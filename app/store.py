"""Platform database: AI cards by key (read-only; deploy-api owns ai_instances) and monthly usage (ai_usage)."""

import ssl
from dataclasses import asdict, dataclass

import asyncpg


@dataclass(frozen=True)
class AIRecord:
    ai_id: str
    user_id: str
    team_id: str | None
    default_chat_model: str | None
    default_embedding_model: str | None
    monthly_budget_paise: int  # 0 = no AI usage allowed

    @property
    def owner(self) -> str:
        return self.team_id or self.user_id

    def to_dict(self) -> dict:
        return asdict(self)


LOOKUP_SQL = """
SELECT a.id::text AS ai_id, a.user_id::text AS user_id, a.team_id::text AS team_id,
       a.default_chat_model, a.default_embedding_model, a.monthly_budget_paise
FROM ai_instances a
WHERE a.token_hash = $1 AND a.status = 'ACTIVE'
"""

CARD_USAGE_SQL = "SELECT requests, tokens_in, tokens_out, cost_mp FROM ai_usage WHERE ai_id = $1::uuid AND period = $2"

UPSERT_USAGE_SQL = """
INSERT INTO ai_usage (ai_id, period, user_id, team_id, requests, tokens_in, tokens_out, cost_mp, updated_at)
VALUES ($1::uuid, $2, $3::uuid, $4::uuid, $5, $6, $7, $8, now())
ON CONFLICT (ai_id, period) DO UPDATE SET
    requests = GREATEST(ai_usage.requests, EXCLUDED.requests),
    tokens_in = GREATEST(ai_usage.tokens_in, EXCLUDED.tokens_in),
    tokens_out = GREATEST(ai_usage.tokens_out, EXCLUDED.tokens_out),
    cost_mp = GREATEST(ai_usage.cost_mp, EXCLUDED.cost_mp),
    updated_at = now()
"""


def asyncpg_dsn(url: str) -> str:
    """Accept the platform's SQLAlchemy URLs (postgresql+pg8000://…) as well as plain DSNs."""
    for prefix in ("postgresql+pg8000://", "postgresql+psycopg://", "postgresql+psycopg2://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    return url


class Store:
    def __init__(self, pool: asyncpg.Pool):
        self.pool = pool

    @classmethod
    async def connect(cls, url: str, ssl_mode: str = "require") -> "Store":
        ssl_arg: object = False
        if ssl_mode != "disable":
            ctx = ssl.create_default_context()
            ctx.check_hostname = False  # same trust model as deploy-api's connection to RDS inside the VPC
            ctx.verify_mode = ssl.CERT_NONE
            ssl_arg = ctx
        pool = await asyncpg.create_pool(asyncpg_dsn(url), min_size=1, max_size=5, ssl=ssl_arg,
                                         command_timeout=5, statement_cache_size=0)
        return cls(pool)

    async def lookup(self, token_hash: str) -> AIRecord | None:
        row = await self.pool.fetchrow(LOOKUP_SQL, token_hash)
        return AIRecord(**dict(row)) if row else None

    async def card_usage(self, ai_id: str, period: str) -> dict[str, int]:
        """{requests, tokens_in, tokens_out, cost_mp} saved for this card and month (zeros if none)."""
        row = await self.pool.fetchrow(CARD_USAGE_SQL, ai_id, period)
        return {k: int(row[k]) if row else 0 for k in ("requests", "tokens_in", "tokens_out", "cost_mp")}

    async def save_usage(self, ai_id: str, period: str, user_id: str, team_id: str | None,
                         requests: int, tokens_in: int, tokens_out: int, cost_mp: int):
        await self.pool.execute(UPSERT_USAGE_SQL, ai_id, period, user_id, team_id,
                                requests, tokens_in, tokens_out, cost_mp)

    async def ping(self):
        await self.pool.fetchval("SELECT 1")

    async def close(self):
        await self.pool.close()
