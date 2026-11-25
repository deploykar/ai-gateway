"""Amazon Bedrock as a model provider: Claude, Nova (chat) and Titan, Cohere (embeddings), in ap-south-1.

Requests arrive in OpenAI's shape and leave the gateway in OpenAI's shape; this module translates to and from
Bedrock's Converse API and the embedding models' own bodies. Credentials come from the pod's IAM role (IRSA).
Nothing Bedrock says (model ids, ARNs, request ids, error text, region) is passed on.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator

from botocore.exceptions import BotoCoreError, ClientError

from app.errors import GatewayError

log = logging.getLogger("ai-gateway")

BUSY = {"ThrottlingException", "ServiceQuotaExceededException", "TooManyRequestsException", "ModelNotReadyException"}
BAD_REQUEST = {"ValidationException"}
UNAVAILABLE = {"AccessDeniedException", "ResourceNotFoundException"}
STOP_REASONS = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "tool_use": "tool_calls",
                "content_filtered": "content_filter", "guardrail_intervened": "content_filter"}
COHERE_BATCH = 96


def bedrock_error(exc: Exception, where: str) -> GatewayError:
    code = exc.response.get("Error", {}).get("Code", "") if isinstance(exc, ClientError) else type(exc).__name__
    log.warning("bedrock error: %s code=%s", where, code)
    if code in BUSY:
        return GatewayError(503, "UPSTREAM_BUSY", "The model is busy right now. Retry in a few seconds.",
                            {"Retry-After": "5"})
    if code in BAD_REQUEST:
        return GatewayError(400, "INVALID_REQUEST", "The model couldn't take this request. Check the messages, "
                                                    "the input size and max_tokens.")
    if code in UNAVAILABLE:
        return GatewayError(503, "MODEL_UNAVAILABLE", "This model isn't available right now. Try another model.")
    return GatewayError(502, "UPSTREAM_ERROR", "The model didn't answer. Retry in a few seconds.")


def _text(content) -> str:
    """OpenAI content: a string, or a list of parts of which only text is used here."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(isinstance(p, dict) and p.get("type") not in (None, "text") for p in content):
            raise GatewayError(400, "INVALID_REQUEST", "Only text messages work with this model for now.")
        return "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def to_converse(body: dict, upstream: str, allow_sampling: bool) -> dict:
    """An OpenAI chat body → Converse arguments."""
    if body.get("tools"):
        raise GatewayError(400, "INVALID_REQUEST", "Tools work with the gpt-oss models for now.")
    system, messages = [], []
    for m in body["messages"]:
        role, text = m.get("role"), _text(m.get("content"))
        if role in ("system", "developer"):
            if text:
                system.append({"text": text})
            continue
        if role not in ("user", "assistant"):
            raise GatewayError(400, "INVALID_REQUEST", f"messages: role {role!r} isn't supported by this model.")
        if not text:
            continue
        if messages and messages[-1]["role"] == role:  # Converse wants turns to alternate
            messages[-1]["content"][0]["text"] += "\n\n" + text
        else:
            messages.append({"role": role, "content": [{"text": text}]})
    if not messages or messages[0]["role"] != "user":
        raise GatewayError(400, "INVALID_REQUEST", "messages: the conversation must start with a user message.")
    config = {"maxTokens": int(body.get("max_tokens") or 1024)}
    if allow_sampling:
        if body.get("temperature") is not None:
            config["temperature"] = float(body["temperature"])
        if body.get("top_p") is not None:
            config["topP"] = float(body["top_p"])
    stop = body.get("stop")
    if stop:
        config["stopSequences"] = [stop] if isinstance(stop, str) else list(stop)[:4]
    args = {"modelId": upstream, "messages": messages, "inferenceConfig": config}
    if system:
        args["system"] = system
    return args


class Bedrock:
    name = "bedrock"
    configured = True

    def __init__(self, client):
        self.client = client  # aiobotocore bedrock-runtime client

    @staticmethod
    def _sampling(upstream: str) -> bool:
        # current Claude models refuse temperature/top_p; Nova takes them
        return "anthropic" not in upstream

    async def chat(self, body: dict) -> dict:
        upstream = body["model"]
        try:
            r = await self.client.converse(**to_converse(body, upstream, self._sampling(upstream)))
        except (ClientError, BotoCoreError) as e:
            raise bedrock_error(e, "converse")
        text = "".join(c.get("text", "") for c in r.get("output", {}).get("message", {}).get("content", []))
        usage = r.get("usage", {})
        return {"choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                             "finish_reason": STOP_REASONS.get(r.get("stopReason"), "stop")}],
                "usage": {"prompt_tokens": usage.get("inputTokens", 0),
                          "completion_tokens": usage.get("outputTokens", 0)}}

    async def chat_stream(self, body: dict) -> AsyncIterator[dict]:
        upstream = body["model"]
        try:
            r = await self.client.converse_stream(**to_converse(body, upstream, self._sampling(upstream)))
        except (ClientError, BotoCoreError) as e:
            raise bedrock_error(e, "converse_stream")
        return self._chunks(r["stream"])

    async def _chunks(self, stream) -> AsyncIterator[dict]:
        try:
            async for event in stream:
                if "contentBlockDelta" in event:
                    text = event["contentBlockDelta"].get("delta", {}).get("text")
                    if text:
                        yield {"choices": [{"index": 0, "delta": {"content": text}}]}
                elif "messageStop" in event:
                    reason = STOP_REASONS.get(event["messageStop"].get("stopReason"), "stop")
                    yield {"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]}
                elif "metadata" in event:
                    u = event["metadata"].get("usage", {})
                    yield {"choices": [], "usage": {"prompt_tokens": u.get("inputTokens", 0),
                                                    "completion_tokens": u.get("outputTokens", 0)}}
        except (ClientError, BotoCoreError) as e:  # mid-stream: the answer just ends
            log.warning("bedrock stream ended early: %s", type(e).__name__)

    async def _invoke(self, upstream: str, payload: dict) -> dict:
        try:
            r = await self.client.invoke_model(modelId=upstream, body=json.dumps(payload),
                                               contentType="application/json", accept="application/json")
            return json.loads(await r["body"].read())
        except (ClientError, BotoCoreError) as e:
            raise bedrock_error(e, "invoke_model")

    async def embeddings(self, body: dict) -> dict:
        upstream, texts = body["model"], body["input"]
        if upstream.startswith("cohere."):
            input_type = body.get("input_type") or "search_document"
            vectors, tokens = [], 0
            for i in range(0, len(texts), COHERE_BATCH):
                part = texts[i:i + COHERE_BATCH]
                out = await self._invoke(upstream, {"texts": part, "input_type": input_type, "truncate": "END"})
                vectors += out["embeddings"]
            return {"data": [{"index": i, "embedding": v} for i, v in enumerate(vectors)]}  # tokens: estimated
        # Titan: one text per call, a few at a time
        dims = int(body.get("dimensions") or 1024)
        sem = asyncio.Semaphore(8)

        async def one(text):
            async with sem:
                return await self._invoke(upstream, {"inputText": text, "dimensions": dims, "normalize": True})
        outs = await asyncio.gather(*(one(t) for t in texts))
        return {"data": [{"index": i, "embedding": o["embedding"]} for i, o in enumerate(outs)],
                "usage": {"prompt_tokens": sum(int(o.get("inputTextTokenCount", 0)) for o in outs)}}
