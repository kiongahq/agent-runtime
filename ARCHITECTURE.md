# Architecture

This document describes how the agent runtime and the two workers that share its
image are organized, and how they connect to the rest of Kionga.

## Bird's-eye view

One image, `ghcr.io/kiongahq/agent-runtime`, holds four top-level Python packages.
The entrypoint decides what runs:

| Entrypoint | Runs as | What it does |
| --- | --- | --- |
| `python -m agent_runtime` (default) | `agent-runtime` service, or one pod per agent on Kubernetes | Serves an agent over HTTP on port `9000` |
| `python -m agent_runtime.designer` | `agent-designer-runner` service | Runs native Agent Designer drafts for the console playground on port `9100` |
| `python -m features.materialize` | `feature-materializer` one-shot job | Loads feature views into the online and offline stores |
| `python -m realtime.service` | `realtime-processor` service | Scores Kafka events with online features |
| `python -m features.retention` | a scheduled job you run | Applies feature TTLs to online keys in Redis and removes keys of deleted views |

Everything builds on the [Python SDK](https://github.com/kiongahq/sdk-python),
installed from a tagged release (`SDK_VERSION`) or a sibling checkout
(`SDK_SOURCE=local`).

## Code map

### `agent_runtime/`

The agent server.

- `app.py`: `create_app` builds the FastAPI app. On startup it reads
  `MLAIOPS_GRAPH_MODULE`, builds the chat model with `mlaiops_sdk.llm`, picks a
  checkpointer (PostgreSQL when `MLAIOPS_CHECKPOINT_DSN` or `DATABASE_URL` is set,
  in-memory otherwise) and loads the graph. Routes: `GET /healthz`,
  `POST /invoke` (one turn, returns reply, usage, cost and duration) and
  `POST /stream` (the same turn as Server-Sent Events). Agent identity comes from
  the gateway's request headers first and `MLAIOPS_AGENT_ID` and
  `MLAIOPS_AGENT_NAME` second, so one shared runtime can serve several agents.
- `graphs.py`: `resolve_attribute` and `build_graph` load a `module:attribute`
  spec that names a compiled graph, an uncompiled `StateGraph` or a
  `build(model, checkpointer)` factory. An attribute marked with
  `kionga_framework` (by the SDK's `agent_factory`) is a framework adapter that
  owns its own model and session state.
- `reporting.py`: `usage_from_messages` sums token counts from LangChain
  `usage_metadata`; `Usage.cost_usd` applies `MLAIOPS_COST_PER_1K_INPUT` and
  `MLAIOPS_COST_PER_1K_OUTPUT` (zero when unset); `report_session` sends the turn
  summary to the gateway through the SDK's `record_trace`
  (`POST /api/v1/traces`).
- `__main__.py`: starts uvicorn on `MLAIOPS_RUNTIME_HOST` and
  `MLAIOPS_RUNTIME_PORT` (default `9000`).
- `native_graph/`: the interpreter for native Agent Designer graphs.
  `components.v1.json` is a byte-identical copy of the contracts registry (a test
  compares them); `migrate.normalize` upgrades schema version 1 drafts in memory;
  `validate.validate_graph` rejects unknown kinds, bad ports, incompatible types and
  cycles; `canonical.content_sha256` is the cross-language behavior hash;
  `conditions.parse` is the `operator:argument` grammar; `engine.run_graph` runs a
  graph in topological order and yields contract run events with redacted,
  truncated previews; `resources.GatewayResources` makes every resource call
  through the gateway with bounded responses and no redirects.
- `designer.py`: the playground runner (`POST /runs`, newline-delimited JSON
  events), authenticated by `KIONGA_DESIGNER_RUNNER_TOKEN`, with a run-scoped gateway
  credential per request, disconnect cancellation and a concurrency cap.
- `evals/`: the paid eval lane. `eval_customer_support.py` runs the example agent
  over `golden.jsonl` and scores answers with an LLM judge.

### `agents/`

Agent code that the runtime loads. `customer_support/graph.py` is a LangGraph
`StateGraph` (reason, tools, respond) whose tools come from the SDK tool registry;
its feature lookup falls back to a local profile when no feature gateway is
configured, so it runs without infrastructure. `framework_examples.py` holds opt-in
Agno and NOOA factories. `native_graph.py` (`build_native`) serves a
published native Agent Designer version after verifying its content hash.

### `features/`

- `definitions.py`: version-controlled feature views (`customer_profile`,
  `transaction_stats_5m`) as declarative data.
- `materialize.py`: `Materializer` waits for its dependencies, applies each view to
  the gateway (`POST /api/v1/features`), writes online rows through the feature
  gateway (`PUT /internal/v1/features/{view}/{entity}`), uploads an offline
  snapshot through a storage-proxy presigned URL (Parquet, or JSONL without
  pyarrow), and reports lineage to the gateway.
- `point_in_time.py`: `point_in_time_join` builds training data that never uses a
  feature row newer than the label's timestamp.
- `retention.py`: `sweep` sets each view's TTL on online keys
  (`mlaiops:features:<view>:<entity>`) that have no expiry and deletes keys of
  views that are no longer defined; `--dry-run` reports without changing anything.

### `realtime/`

- `kafka.py`: `KafkaRest` and `Consumer`, a client for the Kafka REST proxy
  (`KAFKA_REST_URL`), so the Python side needs only `httpx`.
- `demos.py`: three processors (fraud scoring, call-center transcript analysis,
  recommendations). Each falls back to deterministic rules when no model or agent
  endpoint is configured.
- `service.py`: consumes `mlaiops.transactions`, `mlaiops.callcenter.transcripts`
  and `mlaiops.user.activity`, writes results to `mlaiops.fraud.alerts`,
  `mlaiops.callcenter.insights` and `mlaiops.recs.results`, and reports measured
  throughput and latency to the gateway.
- `produce.py`: sends demo events, for example
  `python -m realtime.produce --demo fraud --count 5`.

### `tests/`

Gate tests for each package, with the `mock` LLM and fake HTTP and Redis clients.

## Invariants

- **Measured, never estimated.** Token counts come from the model's own usage
  metadata, cost from configured rates, and real-time statistics from processed
  events. When a number is not available it is reported as zero or unavailable.
- **Reporting never breaks a turn.** Session reports run as background tasks and
  delivery failures are dropped.
- **No credentials in agent code.** Provider keys, DSNs and the gateway token come
  from the environment.
- **Point-in-time correctness.** Training joins never read feature values from the
  future.

## How it connects to other components

- **platform**: the gateway proxies agent invocations to the runtime with the
  agent's identity headers and receives session reports. The runtime authenticates
  with `MLAIOPS_TOKEN`, which deployments set to the platform's internal service
  token. On Kubernetes, the operator runs one runtime per agent with a trace-proxy
  sidecar.
- **trace-proxy**: `MLAIOPS_LLM_BASE_URL` points the `openai` and
  `openai-compatible` backends at the proxy, which records each LLM call.
- **sdk-python**: provides the LLM factory, tools, tracing, memory, checkpointer
  and framework adapters.
- **workspace**: the Jupyter image copies `agents`, `features` and `realtime` from
  a tagged release so notebooks can run them.
- **deploy**: `compose.yaml` runs `agent-runtime`, `feature-materializer` and
  `realtime-processor` from this image with different entrypoints.

## Cross-cutting concerns

- **SDK pin.** `SDK_VERSION` in the `Dockerfile` and `requirements-dev.txt` must
  name the same SDK release.
- **Two test lanes.** Gate tests are free and deterministic. Evals call a real LLM
  and run before ship; see [CONTRIBUTING.md](CONTRIBUTING.md).
- **Security gates.** CI runs ruff, pytest, pip-audit, bandit and gitleaks through
  the shared Python workflow.
