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
LFX 1.13.x. Build it with `--build-arg RUNTIME_FLAVOR=langflow`, or use the image
published by CI. The Kubernetes operator mounts a selected flow revision at
`/app/flows/flow.json` and sets
`MLAIOPS_GRAPH_MODULE=agents.langflow_adapter:build_langflow`. An adjacent
`kionga-api` helper gives Kionga components a project-scoped gateway identity;
its token is not mounted into the agent process or stored in the flow. The
runtime executes each turn through `lfx run` and reports a complete reply.
The editor workspace can be stopped after publication.

## Documentation

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
