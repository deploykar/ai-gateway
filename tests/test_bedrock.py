"""Bedrock: OpenAI-shaped requests in and out, with a fake bedrock-runtime client."""
import asyncio
import json

import pytest
from botocore.exceptions import ClientError

from app.bedrock import Bedrock, to_converse
from app.errors import GatewayError


class Body:
    def __init__(self, data):
        self.data = data

    async def read(self):
        return json.dumps(self.data).encode()


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.fail_code = None

    def _maybe_fail(self):
        if self.fail_code:
            raise ClientError({"Error": {"Code": self.fail_code, "Message": "arn:aws:bedrock:ap-south-1:3093… nope"}},
                              "Converse")

    async def converse(self, **kw):
        self.calls.append(("converse", kw))
        self._maybe_fail()
        return {"output": {"message": {"content": [{"text": "Within 7 days."}]}}, "stopReason": "end_turn",
                "usage": {"inputTokens": 120, "outputTokens": 8}}

    async def converse_stream(self, **kw):
        self.calls.append(("converse_stream", kw))
        self._maybe_fail()

        async def events():
            yield {"messageStart": {"role": "assistant"}}
            for t in ("Within ", "7 days."):
                yield {"contentBlockDelta": {"delta": {"text": t}}}
            yield {"messageStop": {"stopReason": "max_tokens"}}
            yield {"metadata": {"usage": {"inputTokens": 50, "outputTokens": 4}}}
        return {"stream": events()}

    async def invoke_model(self, modelId, body, contentType, accept):
        payload = json.loads(body)
        self.calls.append(("invoke_model", {"modelId": modelId, **payload}))
        self._maybe_fail()
        if modelId.startswith("cohere."):
            return {"body": Body({"embeddings": [[0.1] * 1024 for _ in payload["texts"]]})}
        return {"body": Body({"embedding": [0.2] * payload["dimensions"], "inputTextTokenCount": 3})}


def run(coro):
    return asyncio.run(coro)


def test_messages_become_converse_turns():
    body = {"messages": [{"role": "system", "content": "Answer from the sources."},
                         {"role": "user", "content": [{"type": "text", "text": "Sources: …"}]},
                         {"role": "user", "content": "Question?"},
                         {"role": "assistant", "content": "Earlier answer"},
                         {"role": "user", "content": "Follow-up?"}],
            "max_tokens": 300, "temperature": 0.2, "stop": "END"}
    args = to_converse(body, "global.anthropic.claude-haiku-5-5", allow_sampling=False)
    assert args["system"] == [{"text": "Answer from the sources."}]
    assert [m["role"] for m in args["messages"]] == ["user", "assistant", "user"]  # two user turns merged
    assert args["messages"][0]["content"][0]["text"] == "Sources: …\n\nQuestion?"
    assert args["inferenceConfig"] == {"maxTokens": 300, "stopSequences": ["END"]}  # no temperature for Claude
    assert to_converse(body, "apac.amazon.nova-lite-v1:0", True)["inferenceConfig"]["temperature"] == 0.2


@pytest.mark.parametrize("bad", [
    {"messages": [{"role": "assistant", "content": "hi"}]},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
    {"messages": [{"role": "user", "content": "hi"}], "tools": [{"type": "function"}]},
    {"messages": [{"role": "tool", "content": "x"}]},
])
def test_requests_bedrock_cannot_take(bad):
    with pytest.raises(GatewayError) as e:
        to_converse(bad, "m", True)
    assert e.value.status == 400


def test_chat_answer_in_openai_shape():
    rt = FakeRuntime()
    out = run(Bedrock(rt).chat({"model": "global.anthropic.claude-haiku-5-5",
                                "messages": [{"role": "user", "content": "hi"}]}))
    assert out["choices"][0]["message"]["content"] == "Within 7 days."
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["usage"] == {"prompt_tokens": 120, "completion_tokens": 8}
    assert rt.calls[0][1]["inferenceConfig"]["maxTokens"] == 1024


def test_stream_in_openai_chunks():
    async def collect():
        stream = await Bedrock(FakeRuntime()).chat_stream({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        return [c async for c in stream]
    chunks = run(collect())
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]) == "Within 7 days."
    assert chunks[-2]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"] == {"prompt_tokens": 50, "completion_tokens": 4}


def test_titan_one_call_per_text_with_its_size():
    rt = FakeRuntime()
    out = run(Bedrock(rt).embeddings({"model": "amazon.titan-embed-text-v2:0", "input": ["a", "b", "c"],
                                      "dimensions": 1024}))
    assert [d["index"] for d in out["data"]] == [0, 1, 2] and len(out["data"][0]["embedding"]) == 1024
    assert out["usage"]["prompt_tokens"] == 9
    assert len(rt.calls) == 3 and rt.calls[0][1]["normalize"] is True


def test_cohere_batches_and_input_type():
    rt = FakeRuntime()
    out = run(Bedrock(rt).embeddings({"model": "cohere.embed-multilingual-v3", "input": ["x"] * 100,
                                      "input_type": "search_query"}))
    assert len(out["data"]) == 100
    assert [len(c[1]["texts"]) for c in rt.calls] == [96, 4] and rt.calls[0][1]["input_type"] == "search_query"


@pytest.mark.parametrize("code, status, ours", [
    ("ThrottlingException", 503, "UPSTREAM_BUSY"), ("ValidationException", 400, "INVALID_REQUEST"),
    ("AccessDeniedException", 503, "MODEL_UNAVAILABLE"), ("InternalServerException", 502, "UPSTREAM_ERROR"),
])
def test_errors_are_ours_and_hide_aws(code, status, ours):
    rt = FakeRuntime()
    rt.fail_code = code
    with pytest.raises(GatewayError) as e:
        run(Bedrock(rt).chat({"model": "m", "messages": [{"role": "user", "content": "hi"}]}))
    assert (e.value.status, e.value.code) == (status, ours)
    assert "arn" not in e.value.message and "aws" not in e.value.message.lower()
