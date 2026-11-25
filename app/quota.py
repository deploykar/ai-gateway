"""Redis-backed key cache, per-card rate limit, monthly budget and usage counters.

Keys (deploy-api drops aigw:tok:* when a key is rotated or a card deleted, and reads aigw:use:* for the card):
  aigw:tok:{token_hash}          cached AIRecord JSON ("null" = unknown key)
  aigw:rl:{ai_id}:{minute}       requests in that minute
  aigw:use:{ai_id}:{period}      hash: requests, tokens_in, tokens_out, cost_mp (thousandths of a paisa)
  aigw:dirty                     set of "ai_id|period|user|team" waiting to be saved to Postgres

If Redis is down the gateway keeps serving (keys straight from Postgres; the rate limit is skipped; the budget is
checked against Postgres) and logs it.
"""

import json
import logging
import time
from datetime import datetime, timezone

from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.errors import GatewayError, unauthorized
from app.store import AIRecord, Store

log = logging.getLogger("ai-gateway")
USAGE_TTL_SECONDS = 40 * 86400
FIELDS = ("requests", "tokens_in", "tokens_out", "cost_mp")


def period_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def rupees(paise: int) -> str:
    return f"₹{paise / 100:,.2f}".replace(".00", "")


class Quota:
    def __init__(self, redis: Redis, store: Store, token_ttl: int, unknown_ttl: int, per_minute: int):
        self.redis, self.store = redis, store
        self.token_ttl, self.unknown_ttl, self.per_minute = token_ttl, unknown_ttl, per_minute

    async def resolve(self, token_hash: str) -> AIRecord:
        key = f"aigw:tok:{token_hash}"
        try:
            cached = await self.redis.get(key)
        except RedisError as e:
            log.warning("redis unavailable for key lookup: %s", e)
            cached = None
        if cached is not None:
            data = json.loads(cached)
            if data is None:
                raise unauthorized()
            return AIRecord(**data)
        record = await self.store.lookup(token_hash)
        try:
            await self.redis.set(key, json.dumps(record.to_dict()) if record else "null",
                                 ex=self.token_ttl if record else self.unknown_ttl)
        except RedisError:
            pass
        if record is None:
            raise unauthorized()
        return record

    async def check_rate(self, record: AIRecord):
        key = f"aigw:rl:{record.ai_id}:{int(time.time() // 60)}"
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, 120)
            count, _ = await pipe.execute()
        except RedisError as e:
            log.warning("rate limit skipped (redis): %s", e)
            return
        if count > self.per_minute:
            raise GatewayError(429, "RATE_LIMITED",
                               f"Slow down: an AI card allows {self.per_minute} requests a minute.",
                               {"Retry-After": str(60 - int(time.time()) % 60)})

    async def spent_mp(self, record: AIRecord, period: str) -> int:
        """This card's spend this month, in thousandths of a paisa."""
        key = f"aigw:use:{record.ai_id}:{period}"
        try:
            value = await self.redis.hget(key, "cost_mp")
            if value is None:
                # first request this month on this Redis (or Redis was reset): start from Postgres
                baseline = await self.store.card_usage(record.ai_id, period)
                for field, number in baseline.items():
                    await self.redis.hsetnx(key, field, number)
                await self.redis.expire(key, USAGE_TTL_SECONDS)
                value = await self.redis.hget(key, "cost_mp")
            return int(value or 0)
        except RedisError as e:
            log.warning("budget read from Postgres (redis): %s", e)
            return (await self.store.card_usage(record.ai_id, period))["cost_mp"]

    async def check_budget(self, record: AIRecord):
        budget_mp = record.monthly_budget_paise * 1_000
        if budget_mp <= 0:
            raise GatewayError(429, "BUDGET_EXCEEDED",
                               "This AI card has no monthly budget. Set one on the AI card in DeployKar.")
        if await self.spent_mp(record, period_now()) >= budget_mp:
            raise GatewayError(429, "BUDGET_EXCEEDED",
                               f"This AI card has used its {rupees(record.monthly_budget_paise)} budget for this "
                               f"month. Raise the budget on the AI card in DeployKar, or wait until next month.")

    async def record_usage(self, record: AIRecord, tokens_in: int, tokens_out: int, cost_mp: int):
        period = period_now()
        key = f"aigw:use:{record.ai_id}:{period}"
        try:
            await self.spent_mp(record, period)  # make sure the month starts from Postgres, not from zero
            pipe = self.redis.pipeline()
            pipe.hincrby(key, "requests", 1)
            pipe.hincrby(key, "tokens_in", tokens_in)
            pipe.hincrby(key, "tokens_out", tokens_out)
            pipe.hincrby(key, "cost_mp", cost_mp)
            pipe.expire(key, USAGE_TTL_SECONDS)
            pipe.sadd("aigw:dirty", f"{record.ai_id}|{period}|{record.user_id}|{record.team_id or ''}")
            await pipe.execute()
        except RedisError as e:
            log.error("usage not recorded (redis): ai=%s cost_mp=%s err=%s", record.ai_id, cost_mp, e)

    async def flush_usage(self, batch: int = 500) -> int:
        """Copy per-card counters into Postgres. Safe with many replicas (SPOP + GREATEST upsert)."""
        members = await self.redis.spop("aigw:dirty", batch) or []
        saved = 0
        for member in members:
            ai_id, period, user_id, team_id = member.split("|")
            values = await self.redis.hgetall(f"aigw:use:{ai_id}:{period}")
            if not values:
                continue
            try:
                await self.store.save_usage(ai_id, period, user_id, team_id or None,
                                            *(int(values.get(f, 0)) for f in FIELDS))
                saved += 1
            except Exception as e:  # put it back; the next flush retries
                log.error("usage flush failed for %s: %s", ai_id, e)
                await self.redis.sadd("aigw:dirty", member)
        return saved
