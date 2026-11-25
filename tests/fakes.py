"""In-memory stand-ins for the platform database and a model provider (async, like the real ones)."""

from app.errors import GatewayError
from app.store import AIRecord


class FakeStore:
    def __init__(self):
        self.cards: dict[str, AIRecord] = {}
        self.usage: dict[tuple[str, str], dict] = {}
        self.lookups = 0

    async def lookup(self, token_hash):
        self.lookups += 1
        return self.cards.get(token_hash)

    async def card_usage(self, ai_id, period):
        return dict(self.usage.get((ai_id, period), {"requests": 0, "tokens_in": 0, "tokens_out": 0, "cost_mp": 0}))

    async def save_usage(self, ai_id, period, user_id, team_id, requests, tokens_in, tokens_out, cost_mp):
        old = self.usage.get((ai_id, period), {})
        new = {"requests": requests, "tokens_in": tokens_in, "tokens_out": tokens_out, "cost_mp": cost_mp}
        self.usage[(ai_id, period)] = {k: max(v, old.get(k, 0)) for k, v in new.items()}

    async def ping(self):
        return 1


class FakeProvider:
    """Answers like Cloudflare's OpenAI-compatible endpoints, and records what it was sent."""

    configured = True

    def __init__(self):
        self.sent: list[tuple[str, dict]] = []
        self.fail: GatewayError | None = None
        self.report_usage = True
        self.dims = 768

    def _check(self):
        if self.fail:
            raise self.fail

    async def chat(self, body):
        self.sent.append(("chat", body))
        self._check()
        reply = {"id": "id-cf-123", "object": "chat.completion", "model": body["model"],
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": "Within 7 days [1]."},
                              "finish_reason": "stop"}]}
        if self.report_usage:
            reply["usage"] = {"prompt_tokens": 3000, "completion_tokens": 400, "total_tokens": 3400}
        return reply

    async def chat_stream(self, body):
        self.sent.append(("stream", body))
        self._check()

        async def chunks():
            for word in ("Within ", "7 ", "days."):
                yield {"id": "id-cf-9", "model": body["model"], "system_fingerprint": "cf",
                       "choices": [{"index": 0, "delta": {"content": word}}]}
            if self.report_usage:
                yield {"id": "id-cf-9", "model": body["model"], "choices": [],
                       "usage": {"prompt_tokens": 100, "completion_tokens": 3}}
        return chunks()

    async def embeddings(self, body):
        self.sent.append(("embeddings", body))
        self._check()
        reply = {"object": "list", "model": body["model"],
                 "data": [{"object": "embedding", "index": i, "embedding": [0.5] * self.dims}
                          for i, _ in reversed(list(enumerate(body["input"])))]}  # out of order, like some APIs
        if self.report_usage:
            reply["usage"] = {"prompt_tokens": 10 * len(body["input"]), "total_tokens": 10 * len(body["input"])}
        return reply
