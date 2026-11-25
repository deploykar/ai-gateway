"""Calls to the model providers. Today: Cloudflare Workers AI (OpenAI-compatible endpoints).

Nothing a provider says reaches customers as-is: errors become our own codes, and replies are rebuilt by the
gateway (model name, id). The provider's text is logged only by status and code, never with prompt content.
"""

import json
import logging
from collections.abc import AsyncIterator

import httpx

from app.errors import GatewayError

log = logging.getLogger("ai-gateway")


def upstream_error(status: int, where: str) -> GatewayError:
    log.warning("provider error: %s status=%s", where, status)
    if status == 429:
        return GatewayError(503, "UPSTREAM_BUSY", "The model is busy right now. Retry in a few seconds.",
                            {"Retry-After": "5"})
    if status in (400, 413, 422):
        return GatewayError(400, "INVALID_REQUEST",
                            "The model couldn't take this request. Check the messages, the input size and "
                            "max_tokens.")
    return GatewayError(502, "UPSTREAM_ERROR", "The model didn't answer. Retry in a few seconds.")


class Cloudflare:
    name = "cloudflare"

    def __init__(self, account_id: str, token: str, timeout: float = 120.0, client: httpx.AsyncClient | None = None):
        self.base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        self.token = token
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0))

    @property
    def configured(self) -> bool:
        return bool(self.token) and "/accounts//" not in self.base

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}

    async def _post(self, path: str, body: dict) -> dict:
        try:
            r = await self.client.post(self.base + path, json=body, headers=self._headers())
        except httpx.HTTPError as e:
            log.warning("provider unreachable: %s %s", path, type(e).__name__)
            raise GatewayError(502, "UPSTREAM_ERROR", "The model didn't answer. Retry in a few seconds.")
        if r.status_code >= 400:
            raise upstream_error(r.status_code, path)
        return r.json()

    async def chat(self, body: dict) -> dict:
        return await self._post("/chat/completions", body)

    async def embeddings(self, body: dict) -> dict:
        return await self._post("/embeddings", body)

    async def chat_stream(self, body: dict) -> AsyncIterator[dict]:
        """Yields each streamed chunk as a dict. Raises GatewayError before the first chunk if refused."""
        request = self.client.build_request("POST", self.base + "/chat/completions", json=body,
                                            headers=self._headers())
        try:
            response = await self.client.send(request, stream=True)
        except httpx.HTTPError as e:
            log.warning("provider unreachable (stream): %s", type(e).__name__)
            raise GatewayError(502, "UPSTREAM_ERROR", "The model didn't answer. Retry in a few seconds.")
        if response.status_code >= 400:
            await response.aclose()
            raise upstream_error(response.status_code, "/chat/completions (stream)")
        return self._chunks(response)

    async def _chunks(self, response: httpx.Response) -> AsyncIterator[dict]:
        try:
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    yield json.loads(data)
                except ValueError:
                    continue
        finally:
            await response.aclose()

    async def aclose(self):
        await self.client.aclose()
