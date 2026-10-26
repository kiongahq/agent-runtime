# agent-runtime

[![CI](https://github.com/kiongahq/agent-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/kiongahq/agent-runtime/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENCE)

Serves Kionga AI agents over HTTP, and runs the feature materializer and the real-time processor.

## Background

Kionga is a self-hosted platform for machine learning, data and AI agents. When
you deploy an agent, it runs in this runtime: a FastAPI service that loads your
LangGraph graph (or an Agno, NOOA or custom adapter), answers each turn over HTTP,
keeps session checkpoints, and reports turns, tokens and cost to the Kionga
gateway so you see them in the console.

The same image, `ghcr.io/kiongahq/agent-runtime`, also runs two platform workers:
the feature materializer, which loads feature views into the online and offline
stores, and the real-time processor, which scores Kafka events with online
features. All of them build on the [Python SDK](https://github.com/kiongahq/sdk-python).

## Install

```bash
git clone git@github.com:kiongahq/agent-runtime.git
cd agent-runtime
docker build -t ghcr.io/kiongahq/agent-runtime:dev .
```

For local development without Docker, install the dependencies with
`pip install -r requirements-dev.txt` (Python 3.11).

## Usage

Serve the example customer support agent with the deterministic mock LLM:

```bash
MLAIOPS_GRAPH_MODULE=agents.customer_support.graph:build \
MLAIOPS_LLM_BACKEND=mock \
python -m agent_runtime                       # listens on :9000

curl -X POST http://localhost:9000/invoke \
  -H 'Content-Type: application/json' \
  -d '{"message": "What plan is customer C-1001 on?", "session_id": "demo-1"}'
```

The response holds the reply, the session ID, token counts, cost and duration.
`POST /stream` returns the same turn as Server-Sent Events, and `GET /healthz`
reports the agent's identity. The workers start with
`python -m features.materialize` and `python -m realtime.service`.

### Published Langflow flows

The `langflow-agent-runtime` image uses the same HTTP and reporting contract with
LFX 1.13.x and pinned OpenAI, OpenAI-compatible, Anthropic, and Ollama extensions.
Build it with `--build-arg RUNTIME_FLAVOR=langflow`, or use the image
published by CI. The Kubernetes operator mounts a selected flow revision at
`/app/flows/flow.json` and sets
`MLAIOPS_GRAPH_MODULE=agents.langflow_adapter:build_langflow`. An adjacent
`kionga-api` helper gives Kionga components a project-scoped gateway identity;
its token is not mounted into the agent process or stored in the flow. The
runtime executes each turn through `lfx run` and reports a complete reply.
The editor workspace can be stopped after publication.
Flows that use other provider bundles require a runtime image extended with
those bundles before publishing.

### Native Agent Designer graphs

Graphs built in the console's native Agent Designer are data, not code. This
image interprets them with fixed node handlers (`agent_runtime/native_graph/`)
that follow the [agent graph contract](https://github.com/kiongahq/contracts/tree/main/agent-graph):
typed ports, a deterministic condition grammar, per-node timeouts and retries,
and run budgets (`max_steps`, `timeout_seconds`, `max_tokens`). A node with
inbound edges runs when at least one of them carried a value; a node with no
inbound edges (the input, or a source such as a dataset loader) always runs.
Every resource call (models, tools, API connections, knowledge bases, datasets)
goes through the Kionga gateway, which checks project access on each call and
adds credentials server-side; no secret ever enters a graph or this process.

**Playground runner.** The gateway runs drafts on a separate service:

```bash
KIONGA_DESIGNER_RUNNER_TOKEN=$(openssl rand -hex 24) MLAIOPS_LLM_BACKEND=mock \
python -m agent_runtime.designer              # listens on :9100
```

`POST /runs` (bearer `KIONGA_DESIGNER_RUNNER_TOKEN`) takes
`{run_id, graph, input, session_id, history, allow_side_effects, stop_after, gateway: {url, run_token}}`
and streams run events as newline-delimited JSON. Resource calls go back to
`gateway.url` with `X-Kionga-Run-ID` and `X-Kionga-Run-Token`, so cancelling a run
revokes its access. Tool and API nodes fail with `side_effects_not_approved` unless
`allow_side_effects` is true. `stop_after` tests a single node: the runner executes
only that node and its ancestors. A client disconnect cancels the run, and more than
`KIONGA_DESIGNER_MAX_RUNS` (default 8) concurrent runs return 429. The service
refuses to start without a runner token of at least 16 characters.

**Published versions.** The operator mounts an immutable version artifact at
`/app/flows/flow.json` (override with `KIONGA_NATIVE_GRAPH_PATH`) and sets
`MLAIOPS_GRAPH_MODULE=agents.native_graph:build_native`. The runtime recomputes the
artifact's canonical behavior hash and refuses to start when it does not match, then
serves `/invoke` and `/stream` like any other agent. Resource nodes call the
adjacent loopback `kionga-api` helper (`127.0.0.1:8890`), which holds the agent's
project-scoped identity. Session turns are kept in memory (1,000 sessions, 50 turns
each) for memory nodes.


- Product documentation: [Build agents](https://kiongahq.github.io/docs/guides/agents/),
  [Agent frameworks](https://kiongahq.github.io/docs/guides/agent-frameworks/),
  [Features](https://kiongahq.github.io/docs/guides/features/) and
  [Real-time processing](https://kiongahq.github.io/docs/guides/realtime/)
- Engineering handbook: [AI and observability](https://kiongahq.github.io/docs/engineering/components/ai-observability/)
- How this repository is laid out: [ARCHITECTURE.md](ARCHITECTURE.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the checks CI
runs, agent evals and the commit conventions.

## License

[MIT](LICENCE) © Kionga contributors
