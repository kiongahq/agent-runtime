# agent-runtime

The runtime that serves [Kionga](https://github.com/kiongahq) agents, plus the
Python workers that share its image.

| Package | Runs as | Purpose |
| --- | --- | --- |
| [`agent_runtime`](agent_runtime/README.md) | `agent-runtime` service, one container per deployed agent | Serves a LangGraph agent, or an Agno/NOOA/custom adapter, over HTTP (`/invoke`, `/stream`); checkpoints, memory, session reporting and Langfuse traces; agent evals |
| `agents` | loaded by `agent_runtime` | Example agents (customer support) and framework examples |
| `features` | `feature-materializer` job | Materializes feature definitions to the online store; point-in-time joins; TTL/retention sweeps |
| `realtime` | `realtime-processor` service | Kafka stream processing demos (fraud, call center, recommendations) |

All four depend on the [Python SDK](https://github.com/kiongahq/sdk-python). The image
is `ghcr.io/kiongahq/agent-runtime`; the default entrypoint is `python -m agent_runtime`,
and the other workers override it (`python -m features.materialize`,
`python -m realtime.service`). The [deploy](https://github.com/kiongahq/deploy) repo
wires all three.

## Build

```bash
# SDK from its tagged release (what CI publishes)
docker build -t ghcr.io/kiongahq/agent-runtime .

# SDK from a sibling checkout, for changes that span both repos
docker build --build-arg SDK_SOURCE=local --build-context sdk=../sdk-python \
  -t ghcr.io/kiongahq/agent-runtime:dev .
```

`SDK_VERSION` in the Dockerfile and the pin in `requirements-dev.txt` name the SDK
release this runtime is tested against; bump both together.

## Develop

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt          # or: pip install -e "../sdk-python[runtime,dev]"
python -m pytest -q && ruff check .
```
