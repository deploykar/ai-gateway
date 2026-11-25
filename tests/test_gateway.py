"""The AI gateway end to end, with a fake database, provider and Redis."""
import array
import base64
import hashlib
import json

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app import catalog
from app.config import Settings
from app.errors import GatewayError
from app.main import create_app
from app.quota import period_now
from app.store import AIRecord
from tests.fakes import FakeProvider, FakeStore

KEY = "dkai_live_" + "a" * 40
CARD = AIRecord(ai_id="11111111-1111-1111-1111-111111111111", user_id="22222222-2222-2222-2222-222222222222",
                team_id=None, default_chat_model=None, default_embedding_model=None, monthly_budget_paise=10_000)


@pytest.fixture
def world():
    store, provider = FakeStore(), FakeProvider()
    store.cards[hashlib.sha256(KEY.encode()).hexdigest()] = CARD
    redis = fakeredis.FakeAsyncRedis(decode_responses=True)
    settings = Settings(DATABASE_URL="postgresql://unused", REQUESTS_PER_MINUTE=50, USAGE_FLUSH_SECONDS=3600)
    app = create_app(settings, store=store, redis=redis, providers={"cloudflare": provider})
    with TestClient(app, base_url="https://ai.deploykar.tech") as client:
        client.store, client.provider, client.redis, client.app_ = store, provider, redis, app
        yield client


AUTH = {"Authorization": f"Bearer {KEY}"}


def chat(client, **body):
    body.setdefault("messages", [{"role": "user", "content": "What is the refund window?"}])
    return client.post("/v1/chat/completions", json=body, headers=AUTH)


def usage(client) -> dict:
    """This card's counters for the month, as Redis holds them."""
    return client.portal.call(client.redis.hgetall, f"aigw:use:{CARD.ai_id}:{period_now()}")


# ---------------------------------------------------------------- keys

def test_keys_are_checked(world):
    assert world.post("/v1/chat/completions", json={}).status_code == 401
    r = world.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer dkai_live_wrong"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "UNAUTHORIZED"
    assert world.post("/v1/chat/completions", json={}, headers={"Authorization": "Bearer sk-openai"}).status_code == 401


def test_known_key_is_cached(world):
    chat(world)
    chat(world)
    assert world.store.lookups == 1


# ---------------------------------------------------------------- chat

def test_chat_hides_the_provider_and_counts_the_cost(world):
    r = chat(world, model="fast")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "gpt-oss-20b" and body["id"].startswith("chatcmpl-dk")
    assert body["choices"][0]["message"]["content"] == "Within 7 days [1]."
    assert body["usage"] == {"prompt_tokens": 3000, "completion_tokens": 400, "total_tokens": 3400}
    assert "@cf" not in r.text and "cloudflare" not in r.text.lower()

    kind, sent = world.provider.sent[-1]
    assert sent["model"] == "@cf/openai/gpt-oss-20b"
    # a reasoning model gets room to think and a light effort, so it never answers with nothing
    assert sent["max_tokens"] == catalog.REASONING_MAX_TOKENS and sent["reasoning_effort"] == "low"

    used = usage(world)
    model = catalog.MODELS["gpt-oss-20b"]
    assert int(used["cost_mp"]) == catalog.cost_millipaise(model, 3000, 400) == 3570 + 1424
    assert used["requests"] == "1" and used["tokens_in"] == "3000" and used["tokens_out"] == "400"


def test_only_known_fields_reach_the_model(world):
    chat(world, model="smart", max_completion_tokens=50000, temperature=0.1, api_base="http://evil", user="x")
    _, sent = world.provider.sent[-1]
    assert sent["max_tokens"] == 8000  # capped
    assert "api_base" not in sent and "user" not in sent and sent["temperature"] == 0.1
    assert sent["model"] == "@cf/openai/gpt-oss-120b"


def test_card_default_and_unknown_models(world):
    world.store.cards[hashlib.sha256(KEY.encode()).hexdigest()] = AIRecord(**{**CARD.to_dict(),
                                                                              "default_chat_model": "gpt-oss-120b"})
    world.portal.call(world.redis.flushall)  # forget the cached card
    chat(world, model="default")
    assert world.provider.sent[-1][1]["model"] == "@cf/openai/gpt-oss-120b"
    r = chat(world, model="gpt-4o")
    assert r.status_code == 400 and r.json()["error"]["code"] == "MODEL_NOT_FOUND" and "gpt-oss-20b" in r.text
    r = chat(world, model="bge-m3")  # an embedding model can't chat
    assert r.status_code == 400


def test_bad_requests(world):
    assert chat(world, messages=[]).status_code == 400
    assert chat(world, messages="hi").status_code == 400
    assert world.post("/v1/chat/completions", content=b"not json", headers=AUTH).status_code == 400
    assert chat(world, max_tokens=-5).status_code == 400


def test_usage_is_estimated_when_the_provider_does_not_say(world):
    world.provider.report_usage = False
    body = chat(world).json()
    assert body["usage"]["prompt_tokens"] > 0 and body["usage"]["completion_tokens"] > 0


def test_provider_errors_are_our_own(world):
    world.provider.fail = GatewayError(503, "UPSTREAM_BUSY", "The model is busy right now.")
    r = chat(world)
    assert r.status_code == 503 and r.json()["error"]["code"] == "UPSTREAM_BUSY"
    assert usage(world).get("cost_mp", "0") == "0" and usage(world).get("requests", "0") == "0"  # nothing charged


# ---------------------------------------------------------------- streaming

def test_streaming_rewrites_chunks_and_charges(world):
    with world.stream("POST", "/v1/chat/completions", headers=AUTH,
                      json={"model": "fast", "stream": True, "messages": [{"role": "user", "content": "hi"}]}) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(line[6:]) for line in lines[:-1]]
    assert "".join(c["choices"][0]["delta"]["content"] for c in chunks if c["choices"]) == "Within 7 days."
    assert all(c["model"] == "gpt-oss-20b" and c["id"].startswith("chatcmpl-dk") for c in chunks)
    assert not any("system_fingerprint" in c for c in chunks)
    assert world.provider.sent[-1][1]["stream_options"] == {"include_usage": True}
    used = usage(world)
    assert used["tokens_in"] == "100" and used["tokens_out"] == "3"


# ---------------------------------------------------------------- embeddings

def test_embeddings_in_order_and_charged(world):
    r = world.post("/v1/embeddings", headers=AUTH, json={"model": "embed", "input": ["refunds", "shipping"]})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "bge-base-en-v1.5" and [d["index"] for d in body["data"]] == [0, 1]
    assert len(body["data"][0]["embedding"]) == 768
    assert world.provider.sent[-1][1] == {"model": "@cf/baai/bge-base-en-v1.5", "input": ["refunds", "shipping"]}
    assert usage(world)["tokens_in"] == "20"


def test_embeddings_base64_like_the_openai_sdk_asks(world):
    r = world.post("/v1/embeddings", headers=AUTH, json={"input": "hi", "encoding_format": "base64"})
    vector = array.array("f", base64.b64decode(r.json()["data"][0]["embedding"]))
    assert len(vector) == 768 and abs(vector[0] - 0.5) < 1e-6


def test_embedding_limits(world):
    for bad in ({"input": []}, {"input": [""]}, {"input": [1, 2]}, {"input": ["x"] * 257},
                {"input": ["x" * 50_001]}, {"input": "hi", "encoding_format": "int8"}, {"input": "hi", "model": "fast"}):
        assert world.post("/v1/embeddings", headers=AUTH, json=bad).status_code == 400, bad


# ---------------------------------------------------------------- budget and rate

def test_budget_stops_spending(world):
    world.portal.call(world.redis.hset, f"aigw:use:{CARD.ai_id}:{period_now()}", "cost_mp", 10_000 * 1000)
    r = chat(world)
    assert r.status_code == 429 and r.json()["error"]["code"] == "BUDGET_EXCEEDED" and "₹100" in r.text
    assert world.provider.sent == []  # the model was never called


def test_budget_starts_from_what_postgres_saved(world):
    world.store.usage[(CARD.ai_id, period_now())] = {"requests": 9, "tokens_in": 1, "tokens_out": 1,
                                                       "cost_mp": 10_000 * 1000}
    assert chat(world).status_code == 429


def test_no_budget_means_no_usage(world):
    world.store.cards[hashlib.sha256(KEY.encode()).hexdigest()] = AIRecord(**{**CARD.to_dict(),
                                                                              "monthly_budget_paise": 0})
    world.portal.call(world.redis.flushall)
    assert chat(world).json()["error"]["code"] == "BUDGET_EXCEEDED"


def test_rate_limit(world):
    codes = [chat(world).status_code for _ in range(52)]
    assert codes[:50] == [200] * 50 and codes[-1] == 429


# ---------------------------------------------------------------- usage reaches Postgres

def test_flush_saves_usage(world):
    chat(world)
    chat(world)
    saved = world.portal.call(world.app_.state.quota.flush_usage)
    assert saved == 1
    row = world.store.usage[(CARD.ai_id, period_now())]
    assert row["requests"] == 2 and row["tokens_in"] == 6000


# ---------------------------------------------------------------- the rest

def test_models_list(world):
    assert world.get("/v1/models").status_code == 401
    ids = [m["id"] for m in world.get("/v1/models", headers=AUTH).json()["data"]]
    assert {"gpt-oss-20b", "gpt-oss-120b", "bge-base-en-v1.5", "bge-m3", "fast", "smart", "embed"} <= set(ids)


def test_health_only_inside_the_cluster(world):
    assert world.get("/healthz").status_code == 404  # the public address
    assert world.get("/healthz", headers={"host": "ai-gateway.deploykar-system"}).json() == {"status": "ok"}


def test_large_bodies_refused(world):
    r = world.post("/v1/chat/completions", headers=AUTH, content=b"x" * 2_000_001)
    assert r.status_code == 413
