# ai-gateway

The API behind DeployKar's **AI card**. Apps call one OpenAI-compatible address with a `dkai_live_…` key; this
service checks the key, keeps each card inside its rate limit and monthly ₹ budget, sends the request to a
model provider and counts what it cost. Customers see DeployKar's model names and errors only: never a
provider's name, account, region or error text. Prompts and answers are never logged or stored.

```
app ──HTTPS + dkai_ key──► Kong (ai.deploykar.tech) ──► ai-gateway ──► Cloudflare Workers AI (open models)
                                                           │            (Bedrock: Claude, Nova, Titan, later)
                                         Redis: key cache, rate limit, usage counters
                                         Postgres: ai_instances (read), ai_usage (write)
```

deploy-api owns the AI card (create, key, budget, default models, wires, delete). The gateway only reads cards.

## API

Base URL `https://ai.deploykar.tech/v1`, header `Authorization: Bearer $DEPLOYKAR_AI_KEY`. The OpenAI SDKs work
as they are: set `OPENAI_BASE_URL` and `OPENAI_API_KEY`.

| Call | What it does |
|---|---|
| `POST /v1/chat/completions` | Chat, also with `"stream": true`. `model` is a name below, an alias, or `default` (the card's choice) |
| `POST /v1/embeddings` | Vectors for a text or a list of up to 256 texts (`encoding_format` float or base64) |
| `GET /v1/models` | Names, aliases, what each is for, price in ₹ per million tokens |

| Name | Alias | Kind |
|---|---|---|
| `gpt-oss-20b` | `fast` | chat (default) |
| `gpt-oss-120b` | `smart` | chat |
| `bge-base-en-v1.5` | `embed` | embeddings, 768 numbers (default) |
| `bge-m3` | `embed-multilingual` | embeddings, 1,024 numbers, many languages |

Reasoning models get `max_tokens` 2000 and `reasoning_effort` low unless the request sets them, so they never
answer with nothing. `max_tokens` is capped at 8,000. Only standard chat fields reach the model.

Errors are `{"error": {"code", "message", "type"}}`: `UNAUTHORIZED` 401, `INVALID_REQUEST` / `MODEL_NOT_FOUND` 400,
`REQUEST_TOO_LARGE` 413, `RATE_LIMITED` / `BUDGET_EXCEEDED` 429, `UPSTREAM_ERROR` 502, `UPSTREAM_BUSY` /
`MODEL_UNAVAILABLE` 503 (retry).

## Settings

`DATABASE_URL`, `REDIS_URL`, `CF_AI_ACCOUNT_ID`, `CF_AI_TOKEN` (from a hand-made Secret), optional
`REQUESTS_PER_MINUTE` (120 per card), `MAX_OUTPUT_TOKENS`, `UPSTREAM_TIMEOUT_SECONDS`.

Prices in `app/catalog.py` are what we charge (provider cost + ~25%) and are approximate until checked against
the provider's price list.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Deploying

Push to `main`: GitHub Actions tests, builds `linux/arm64`, pushes `deploykar/ai-gateway:git-<sha>` and sets that
tag in deploykar-infra `eks/services/ai-gateway`; Argo CD rolls it out within a minute (2 pods on platform nodes).

- AWS: IRSA role `deploykar-prod-ai-gateway` (deploykar-infra `terraform/modules/07-iam/ai_gateway.tf`) may invoke
  only the catalog's Bedrock models. Adding a Bedrock model = add it to that file and to `app/catalog.py`.
- Cloudflare: Secret `deploykar-system/ai-gateway` (`CF_AI_ACCOUNT_ID`, `CF_AI_TOKEN`), made by hand.
- Public address: `ai.deploykar.tech` (DNS only, CNAME to the Kong load balancer; certificate from
  ClusterIssuer `letsencrypt-ai`).
- Who sees AI cards: deploy-api `AI_CARDS` ("" nobody, "*" everyone, or user ids).
