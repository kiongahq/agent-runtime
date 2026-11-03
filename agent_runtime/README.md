# agent-runtime

Serves a compiled LangGraph agent or a marked Agno/NOOA/custom adapter over HTTP.
See [framework integration](https://kiongahq.github.io/docs/guides/agent-frameworks/) for packaging,
provider ownership, buffered streaming, and NOOA safety requirements. This is the container a
`KiongaAgent` runs in production and the `agent-runtime` Compose service
locally.

## Contract

| Endpoint | Purpose |
| --- | --- |
| `GET /healthz` | Liveness, framework and auth mode (no identity; unauthenticated) |
| `POST /invoke` | One agent turn: `{message, session_id?, user_id?}` → reply + usage counters and `usage_available` |
| `POST /stream` | Same turn as Server-Sent Events (`{"delta": ...}` chunks, final `{"done": true, ...}` summary) |

Every turn reports a session summary to the gateway (`POST /api/v1/traces`),
which upserts the live session (turns, tokens, cost) shown in the console.
The gateway attributes these reports to the narrow internal `service` role when
`MLAIOPS_INTERNAL_TOKEN` is configured; do not reuse a user personal key.

## Authentication

Only the gateway may call `/invoke` and `/stream`. With auth required the runtime
expects these headers on every turn:

| Header | Meaning |
| --- | --- |
| `Authorization: Bearer $MLAIOPS_INTERNAL_TOKEN` | Gateway credential, compared in constant time; otherwise 401 |
| `X-MLAIOps-User-ID` | The authenticated end user; required (401 without it). A body `user_id` that differs gets 403 |
| `X-MLAIOps-Agent-ID` / `X-MLAIOps-Agent-Name` | Control-plane agent identity for session reporting |
| `X-MLAIOps-Graph-Module` | Entry point the gateway expects; 409 on mismatch |

The credential check runs before body validation, so an unauthenticated caller
gets 401 and never a schema error. Startup fails when auth is required and
`MLAIOPS_INTERNAL_TOKEN` is unset or shorter than 16 characters, and when
`MLAIOPS_RUNTIME_AUTH=disabled` is combined with `KIONGA_ENVIRONMENT=production`.
With auth disabled (the local default) every header and the body `user_id` are
taken as given, which is safe only on a developer machine.

## Configuration

| Variable | Meaning |
| --- | --- |
| `MLAIOPS_GRAPH_MODULE` | `package.module:attribute` — compiled graph, `StateGraph`, or `build(model, checkpointer)` factory |
| `MLAIOPS_AGENT_ID` / `MLAIOPS_AGENT_NAME` | Control-plane identity for session reporting |
| `MLAIOPS_URL` | Gateway base URL for session reporting |
| `MLAIOPS_INTERNAL_TOKEN` | Shared gateway credential: verified on `/invoke` and `/stream` when auth is required; also the session-reporting credential |
| `MLAIOPS_RUNTIME_AUTH` | `required` or `disabled`; defaults to `required` when `KIONGA_ENVIRONMENT=production`, `disabled` otherwise |
| `KIONGA_ENVIRONMENT` | `production` makes auth required and refuses `MLAIOPS_RUNTIME_AUTH=disabled` |
| `DATABASE_URL` or `MLAIOPS_CHECKPOINT_DSN` | Enables the PostgreSQL LangGraph checkpointer (in-memory otherwise) |
| `MLAIOPS_LLM_BACKEND` / `MLAIOPS_LLM_MODEL` / `MLAIOPS_LLM_BASE_URL` | Chat model selection (`openai`, `anthropic`, `openai-compatible`, `mock`) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | Full-trace delivery to Langfuse |
| `MLAIOPS_COST_PER_1K_INPUT` / `MLAIOPS_COST_PER_1K_OUTPUT` | Deterministic cost calculation (USD per 1k tokens) |

LLM calls should egress through the platform trace-proxy: set
`MLAIOPS_LLM_BASE_URL` (or `OPENAI_BASE_URL`) to the proxy address so every
request/response pair is captured.

Session checkpoints and long-term memory are separate: checkpoints preserve graph
thread state, while `AgentMemoryClient` stores tenant/agent/user-scoped semantic
memories in pgvector. Keep project/user identifiers in runtime context when serving
multiple teams.

## Code execution

The runtime does not execute model-generated code. Frameworks that do, such as
NOOA, run that code inside the runtime process, which serves HTTP and holds
provider keys and `MLAIOPS_INTERNAL_TOKEN`. Generated code there could read those
secrets from `os.environ`, call internal services and exhaust the pod.

Threat model for enabling it later: generated code is untrusted input. It needs
an out-of-process executor (a gVisor or Firecracker runtime class, or a separate
executor service) with no secrets in its environment, a read-only root
filesystem, no network by default, and CPU, memory and wall-clock limits, and the
runtime must verify that executor is configured and healthy before it starts.
Until that exists, `agents.framework_examples:build_nooa` refuses to start
(`require_code_sandbox`). `KIONGA_ALLOW_CODE_EXECUTION` is no longer honored: an
environment flag is not a sandbox.

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
