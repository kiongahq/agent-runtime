# agent-runtime

Serves a compiled LangGraph agent over HTTP. This is the container a
`NexusAgent` runs in production and the `agent-runtime` Compose service
locally.

## Contract

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Liveness + agent identity |
| `POST /invoke` | One agent turn: `{message, session_id?, user_id?}` → reply + exact token usage |
| `POST /stream` | Same turn as Server-Sent Events (`{"delta": ...}` chunks, final `{"done": true, ...}` summary) |

Every turn reports a session summary to the gateway (`POST /api/v1/traces`),
which upserts the live session (turns, tokens, cost) shown in the console.

## Configuration

| Variable | Meaning |
| --- | --- |
| `MLAIOPS_GRAPH_MODULE` | `package.module:attribute` — compiled graph, `StateGraph`, or `build(model, checkpointer)` factory |
| `MLAIOPS_AGENT_ID` / `MLAIOPS_AGENT_NAME` | Control-plane identity for session reporting |
| `MLAIOPS_URL` | Gateway base URL for session reporting |
| `DATABASE_URL` or `MLAIOPS_CHECKPOINT_DSN` | Enables the PostgreSQL LangGraph checkpointer (in-memory otherwise) |
| `MLAIOPS_LLM_BACKEND` / `MLAIOPS_LLM_MODEL` / `MLAIOPS_LLM_BASE_URL` | Chat model selection (`openai`, `anthropic`, `openai-compatible`, `mock`) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Full-trace delivery to Langfuse |
| `MLAIOPS_COST_PER_1K_INPUT` / `MLAIOPS_COST_PER_1K_OUTPUT` | Deterministic cost calculation (USD per 1k tokens) |

LLM calls should egress through the platform trace-proxy: set
`MLAIOPS_LLM_BASE_URL` (or `OPENAI_BASE_URL`) to the proxy address so every
request/response pair is captured.

## Run locally

```bash
MLAIOPS_GRAPH_MODULE=agents.customer_support.graph:build \
MLAIOPS_LLM_BACKEND=mock \
PYTHONPATH=python python -m agent_runtime
```

## Tests and evals

- Gate tests: `python -m pytest python/tests/test_agent_runtime.py` (mock LLM, no network).
- Evals (paid lane): `MLAIOPS_RUN_EVALS=1 python -m pytest python/agent_runtime/evals` — runs
  the demo agent against golden questions with a real LLM and an LLM-as-judge
  scorer; threshold documented in `evals/README.md`.
