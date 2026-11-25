"""The models customers can use, under DeployKar's own names, and what each costs them.

Prices are what WE CHARGE, in paise per million tokens: the provider's cost plus ~25%, at ₹95 per $1.
- Claude and Cohere: Bedrock Marketplace rate cards read 2026-11-25 (global route; Claude Haiku 5.5 $0.10/$0.50,
  Sonnet 5.5 $2/$10, Cohere Embed Multilingual $0.10 per million tokens).
- Nova Lite and Titan Text Embeddings V2: APPROXIMATE ($0.063/$0.252 and $0.02), check AWS's page.
- Cloudflare bills $0.011 per 1,000 neurons; the neuron rates are the ones deploypilot uses (app/assistant/ai.py)
  and Cloudflare's embedding rates as of Nov 2026: APPROXIMATE.
Billing isn't live: today these only fill budgets.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    name: str                # what customers send and see
    kind: str                # "chat" | "embedding"
    provider: str            # "cloudflare" | "bedrock"
    upstream: str            # the provider's id: never shown
    in_paise_per_m: int      # charged per million input tokens
    out_paise_per_m: int = 0
    dims: int | None = None  # embeddings: numbers per vector
    reasoning: bool = False  # thinks before answering: needs room (max_tokens) or it can answer nothing
    blurb: str = ""


MODELS: dict[str, Model] = {m.name: m for m in (
    Model("gpt-oss-20b", "chat", "cloudflare", "@cf/openai/gpt-oss-20b", 1_190, 3_560, reasoning=True,
          blurb="Fast and cheap. Good for most RAG answers and support bots."),
    Model("gpt-oss-120b", "chat", "cloudflare", "@cf/openai/gpt-oss-120b", 4_160, 8_910, reasoning=True,
          blurb="Smarter. Long documents and questions that need reasoning."),
    Model("bge-base-en-v1.5", "embedding", "cloudflare", "@cf/baai/bge-base-en-v1.5", 790, dims=768,
          blurb="English text, 768 numbers per vector."),
    Model("bge-m3", "embedding", "cloudflare", "@cf/baai/bge-m3", 143, dims=1024,
          blurb="Many languages (Hindi, Marathi, Tamil…), 1,024 numbers per vector."),
    # Bedrock (Claude 5.5 runs only on the global route from Mumbai: prompts may be processed outside India)
    Model("claude-haiku-5-5", "chat", "bedrock", "global.anthropic.claude-haiku-5-5", 1_188, 5_938,
          blurb="Claude Haiku 5.5: fast, careful answers that stick to your documents."),
    Model("claude-sonnet-5-5", "chat", "bedrock", "global.anthropic.claude-sonnet-5-5", 23_750, 118_750,
          blurb="Claude Sonnet 5.5: the strongest here, for hard questions and long documents."),
    Model("nova-lite", "chat", "bedrock", "apac.amazon.nova-lite-v1:0", 748, 2_993,
          blurb="Nova Lite: the cheapest chat model here."),
    Model("titan-embed-v2", "embedding", "bedrock", "amazon.titan-embed-text-v2:0", 238, dims=1024,
          blurb="English and 100+ languages, 1,024 numbers per vector; stays in India."),
    Model("cohere-embed-multilingual", "embedding", "bedrock", "cohere.embed-multilingual-v3", 1_188, dims=1024,
          blurb="Mixed Hindi/English and other languages, 1,024 numbers per vector; stays in India."),
)}

ALIASES = {"fast": "gpt-oss-20b", "smart": "gpt-oss-120b", "embed": "bge-base-en-v1.5",
           "embed-multilingual": "bge-m3", "claude": "claude-haiku-5-5"}

DEFAULT_CHAT = "gpt-oss-20b"
DEFAULT_EMBEDDING = "bge-base-en-v1.5"

# reasoning models get these unless the request says otherwise (without them gpt-oss can spend its whole
# output allowance thinking and return an empty answer)
REASONING_MAX_TOKENS = 2_000
REASONING_EFFORT = "low"


def resolve(requested: str | None, kind: str, card_default: str | None) -> Model | None:
    """The model a request means: a name, an alias, or "default"/nothing (the card's choice)."""
    name = (requested or "").strip()
    if name in ("", "default"):
        name = card_default or (DEFAULT_CHAT if kind == "chat" else DEFAULT_EMBEDDING)
    name = ALIASES.get(name, name)
    model = MODELS.get(name)
    return model if model and model.kind == kind else None


def cost_millipaise(model: Model, tokens_in: int, tokens_out: int = 0) -> int:
    """What a request costs the customer, in thousandths of a paisa (rounded up, never below 1)."""
    total = tokens_in * model.in_paise_per_m + tokens_out * model.out_paise_per_m  # paise per M tokens × tokens
    return max(1, -(-total // 1_000)) if total else 0


def public_list() -> list[dict]:
    """GET /v1/models: OpenAI's list shape, plus what a model is for and its price in rupees."""
    out = [{"id": m.name, "object": "model", "owned_by": "deploykar", "type": m.kind, "description": m.blurb,
            "price_inr_per_million_tokens": {"input": m.in_paise_per_m / 100,
                                             **({"output": m.out_paise_per_m / 100} if m.kind == "chat" else {})},
            **({"dimensions": m.dims} if m.dims else {})} for m in MODELS.values()]
    out += [{"id": alias, "object": "model", "owned_by": "deploykar", "alias_of": target}
            for alias, target in ALIASES.items()]
    return out
