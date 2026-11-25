"""DeployKar AI Gateway.

OpenAI-compatible API at https://{AI_DOMAIN}/v1 for the AI card on the canvas: chat completions (streaming too),
embeddings and the model list. Authenticates dkai_live_… keys, enforces each card's rate limit and monthly ₹
budget, and counts usage. Customers see DeployKar's model names and errors only: never a provider's name,
account, region, ids or error text. Prompts and answers are never logged or stored.
"""

import array
import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from redis.asyncio import Redis

from app import catalog
from app.config import Settings, get_settings
from app.errors import (
    GatewayError,
    error_body,
    gateway_error_handler,
    unauthorized,
    unexpected_error_handler,
    validation_error_handler,
)
from app.bedrock import Bedrock
from app.providers import Cloudflare
from app.quota import Quota
from app.store import AIRecord, Store

log = logging.getLogger("ai-gateway")

REQUESTS = Counter("ai_gateway_requests_total", "Requests", ["route", "status"])
LATENCY = Histogram("ai_gateway_request_seconds", "Request latency", ["route"],
                    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40, 80, 120))
TOKENS = Counter("ai_gateway_tokens_total", "Tokens", ["model", "direction"])
TOO_LARGE_BODY = json.dumps(error_body("REQUEST_TOO_LARGE", "Request body is too large.")).encode()

# What a chat request may carry to the model (anything else is dropped, never forwarded)
CHAT_FIELDS = {"messages", "max_tokens", "temperature", "top_p", "stop", "tools", "tool_choice", "response_format",
               "stream", "seed", "frequency_penalty", "presence_penalty", "reasoning_effort"}
MAX_EMBED_ITEMS = 256
MAX_EMBED_CHARS = 50_000


class _BodyTooLarge(Exception):
    pass


class LimitBodyMiddleware:
    """Reject request bodies over max_bytes, whether or not Content-Length is sent (as queue-gateway)."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            return await self._reject(send)
        seen, started = 0, False

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _BodyTooLarge()
            return message

        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _BodyTooLarge:
            if not started:
                await self._reject(send)

    @staticmethod
    async def _reject(send):
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(TOO_LARGE_BODY)).encode())]})
        await send({"type": "http.response.body", "body": TOO_LARGE_BODY})


def estimate_tokens(text: str) -> int:
    """About 4 characters a token: used only when a provider doesn't report usage."""
    return max(1, -(-len(text) // 4)) if text else 0


def _invalid(message: str) -> GatewayError:
    return GatewayError(400, "INVALID_REQUEST", message)


def _model_error(requested, kind: str) -> GatewayError:
    names = ", ".join([m.name for m in catalog.MODELS.values() if m.kind == kind] +
                      [a for a, t in catalog.ALIASES.items() if catalog.MODELS[t].kind == kind])
    what = "chat" if kind == "chat" else "embedding"
    return GatewayError(400, "MODEL_NOT_FOUND", f"No {what} model called {requested!r}. Use one of: {names}, "
                                                f"or \"default\" (the AI card's choice).")


async def _json_body(request: Request) -> dict:
    try:
        body = json.loads(await request.body() or b"{}")
    except ValueError:
        raise _invalid("The request body must be JSON.")
    if not isinstance(body, dict):
        raise _invalid("The request body must be a JSON object.")
    return body


def _float32_base64(vector: list[float]) -> str:
    return base64.b64encode(array.array("f", vector).tobytes()).decode()


def create_app(settings: Settings | None = None, *, store=None, redis=None, providers=None) -> FastAPI:
    settings = settings or get_settings()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        async with contextlib.AsyncExitStack() as stack:
            state = app.state
            state.store = store or await Store.connect(settings.DATABASE_URL, settings.DATABASE_SSL)
            if store is None:
                stack.push_async_callback(state.store.close)
            state.redis = redis or Redis.from_url(settings.REDIS_URL, decode_responses=True,
                                                  socket_connect_timeout=2, socket_timeout=2,
                                                  health_check_interval=30)
            if redis is None:
                stack.push_async_callback(state.redis.aclose)
            if providers is None:
                cf = Cloudflare(settings.CF_AI_ACCOUNT_ID, settings.CF_AI_TOKEN, settings.UPSTREAM_TIMEOUT_SECONDS)
                stack.push_async_callback(cf.aclose)
                state.providers = {"cloudflare": cf}
                if settings.BEDROCK_ENABLED:
                    from aiobotocore.config import AioConfig
                    from aiobotocore.session import get_session
                    config = AioConfig(retries={"mode": "standard", "max_attempts": 2}, connect_timeout=5,
                                       read_timeout=settings.UPSTREAM_TIMEOUT_SECONDS, max_pool_connections=100)
                    client = await stack.enter_async_context(
                        get_session().create_client("bedrock-runtime", region_name=settings.AWS_REGION, config=config))
                    state.providers["bedrock"] = Bedrock(client)
            else:
                state.providers = providers
            state.quota = Quota(state.redis, state.store, settings.TOKEN_CACHE_SECONDS,
                                settings.UNKNOWN_TOKEN_CACHE_SECONDS, settings.REQUESTS_PER_MINUTE)

            async def flush_loop():
                while True:
                    await asyncio.sleep(settings.USAGE_FLUSH_SECONDS)
                    try:
                        await state.quota.flush_usage()
                    except Exception as e:
                        log.error("usage flush loop: %s", e)

            flusher = asyncio.create_task(flush_loop())
            try:
                yield
            finally:
                flusher.cancel()
                with contextlib.suppress(BaseException):
                    await flusher
                with contextlib.suppress(Exception):
                    await state.quota.flush_usage()  # don't lose the last few seconds on shutdown

    app = FastAPI(title="DeployKar AI Gateway", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.add_exception_handler(GatewayError, gateway_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(Exception, unexpected_error_handler)
    app.add_middleware(LimitBodyMiddleware, max_bytes=settings.MAX_REQUEST_BYTES)
    ai_domain = settings.AI_DOMAIN.lower()

    def internal_only(request: Request):
        # health and metrics are for the cluster, not for the public address
        if request.headers.get("host", "").split(":")[0].lower().rstrip(".") == ai_domain:
            raise GatewayError(404, "NOT_FOUND", "Not found.")

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        name = getattr(route, "path", "unmatched")
        elapsed = time.perf_counter() - start
        REQUESTS.labels(name, str(response.status_code)).inc()
        LATENCY.labels(name).observe(elapsed)
        if name not in ("/healthz", "/readyz", "/metrics"):
            # no prompt, answer or key here: only who, which route, and how it went
            log.info("%s %s %s %.0fms ai=%s model=%s", request.method, name, response.status_code, elapsed * 1000,
                     getattr(request.state, "ai_id", "-"), getattr(request.state, "model", "-"))
        return response

    async def card(request: Request) -> AIRecord:
        scheme, _, token = request.headers.get("authorization", "").partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token.startswith("dkai_"):
            raise unauthorized()
        state = request.app.state
        record = await state.quota.resolve(hashlib.sha256(token.encode()).hexdigest())
        request.state.ai_id = record.ai_id
        await state.quota.check_rate(record)
        return record

    def provider_for(request: Request, model: catalog.Model):
        provider = request.app.state.providers.get(model.provider)
        if provider is None or not getattr(provider, "configured", True):
            raise GatewayError(503, "MODEL_UNAVAILABLE", f"{model.name} isn't available right now. Try another "
                                                         f"model, or retry later.")
        return provider

    async def charge(request: Request, record: AIRecord, model: catalog.Model, tokens_in: int, tokens_out: int):
        TOKENS.labels(model.name, "in").inc(tokens_in)
        TOKENS.labels(model.name, "out").inc(tokens_out)
        await request.app.state.quota.record_usage(record, tokens_in, tokens_out,
                                                   catalog.cost_millipaise(model, tokens_in, tokens_out))

    api = APIRouter(prefix="/v1")

    @api.get("/models")
    async def list_models(_: AIRecord = Depends(card)):
        return {"object": "list", "data": catalog.public_list()}

    @api.post("/chat/completions")
    async def chat_completions(request: Request, record: AIRecord = Depends(card)):
        body = await _json_body(request)
        model = catalog.resolve(body.get("model"), "chat", record.default_chat_model)
        if model is None:
            raise _model_error(body.get("model"), "chat")
        request.state.model = model.name
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) and m.get("role")
                                                                      for m in messages):
            raise _invalid("messages: a non-empty list of {role, content} objects is needed.")
        await request.app.state.quota.check_budget(record)
        provider = provider_for(request, model)

        out = {k: v for k, v in body.items() if k in CHAT_FIELDS}
        max_tokens = body.get("max_completion_tokens") or out.get("max_tokens")
        if model.reasoning and not max_tokens:
            max_tokens = catalog.REASONING_MAX_TOKENS
        if max_tokens is not None:
            if not isinstance(max_tokens, int) or max_tokens < 1:
                raise _invalid("max_tokens must be a positive whole number.")
            out["max_tokens"] = min(max_tokens, settings.MAX_OUTPUT_TOKENS)
        if model.reasoning and "reasoning_effort" not in out:
            out["reasoning_effort"] = catalog.REASONING_EFFORT
        out["model"] = model.upstream
        reply_id = f"chatcmpl-dk{uuid.uuid4().hex[:24]}"
        created = int(time.time())
        prompt_chars = len(json.dumps(messages))

        if not out.get("stream"):
            reply = await provider.chat(out)
            choices = reply.get("choices") or []
            usage = reply.get("usage") or {}
            tokens_in = int(usage.get("prompt_tokens") or estimate_tokens("x" * prompt_chars))
            answer = "".join(str((c.get("message") or {}).get("content") or "") for c in choices)
            tokens_out = int(usage.get("completion_tokens") or estimate_tokens(answer))
            await charge(request, record, model, tokens_in, tokens_out)
            return {"id": reply_id, "object": "chat.completion", "created": created, "model": model.name,
                    "choices": choices,
                    "usage": {"prompt_tokens": tokens_in, "completion_tokens": tokens_out,
                              "total_tokens": tokens_in + tokens_out}}

        out["stream_options"] = {"include_usage": True}
        chunks = await provider.chat_stream(out)  # refused before streaming: a normal error response

        async def events():
            usage, text_len = {}, 0
            try:
                async for chunk in chunks:
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for c in chunk.get("choices") or []:
                        text_len += len(str((c.get("delta") or {}).get("content") or ""))
                    chunk.update(id=reply_id, model=model.name, created=created)
                    chunk.pop("system_fingerprint", None)
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            finally:
                tokens_in = int(usage.get("prompt_tokens") or estimate_tokens("x" * prompt_chars))
                tokens_out = int(usage.get("completion_tokens") or estimate_tokens("x" * text_len))
                await charge(request, record, model, tokens_in, tokens_out)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @api.post("/embeddings")
    async def embeddings(request: Request, record: AIRecord = Depends(card)):
        body = await _json_body(request)
        model = catalog.resolve(body.get("model"), "embedding", record.default_embedding_model)
        if model is None:
            raise _model_error(body.get("model"), "embedding")
        request.state.model = model.name
        items = body.get("input")
        items = [items] if isinstance(items, str) else items
        if not isinstance(items, list) or not items or not all(isinstance(t, str) and t for t in items):
            raise _invalid("input: a text, or a list of texts, is needed.")
        if len(items) > MAX_EMBED_ITEMS:
            raise _invalid(f"input: at most {MAX_EMBED_ITEMS} texts in one request.")
        if any(len(t) > MAX_EMBED_CHARS for t in items):
            raise _invalid(f"input: each text can be up to {MAX_EMBED_CHARS:,} characters. Split longer ones.")
        encoding = body.get("encoding_format") or "float"
        if encoding not in ("float", "base64"):
            raise _invalid('encoding_format must be "float" or "base64".')
        await request.app.state.quota.check_budget(record)
        provider = provider_for(request, model)

        request_body = {"model": model.upstream, "input": items}
        if model.dims:
            request_body["dimensions"] = model.dims
        if body.get("input_type") in ("search_document", "search_query"):  # Cohere: documents vs questions
            request_body["input_type"] = body["input_type"]
        reply = await provider.embeddings(request_body)
        rows = sorted(reply.get("data") or [], key=lambda d: d.get("index", 0))
        if len(rows) != len(items):
            raise GatewayError(502, "UPSTREAM_ERROR", "The model didn't answer. Retry in a few seconds.")
        usage = reply.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or sum(estimate_tokens(t) for t in items))
        await charge(request, record, model, tokens_in, 0)
        data = [{"object": "embedding", "index": i,
                 "embedding": _float32_base64(r["embedding"]) if encoding == "base64" else r["embedding"]}
                for i, r in enumerate(rows)]
        return {"object": "list", "data": data, "model": model.name,
                "usage": {"prompt_tokens": tokens_in, "total_tokens": tokens_in}}

    app.include_router(api)

    @app.get("/healthz", dependencies=[Depends(internal_only)])
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz", dependencies=[Depends(internal_only)])
    async def readyz(request: Request):
        state = request.app.state
        try:
            await state.store.ping()
            await state.redis.ping()
        except Exception as e:
            log.warning("not ready: %s", e)
            raise GatewayError(503, "NOT_READY", "Dependencies unavailable.")
        return {"status": "ready"}

    @app.get("/metrics", dependencies=[Depends(internal_only)])
    async def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


def build() -> FastAPI:
    """uvicorn entrypoint: `uvicorn app.main:build --factory`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    return create_app()
