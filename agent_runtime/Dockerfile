# Agent runtime image: serves a LangGraph or marked framework adapter factory.
# Build from the repository root:
#   docker build -f python/agent_runtime/Dockerfile -t mlaiops/agent-runtime .
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

WORKDIR /app

COPY python/pyproject.toml python/pyproject.toml
COPY python/mlaiops_sdk python/mlaiops_sdk
COPY python/agent_runtime python/agent_runtime
COPY python/agents python/agents
COPY python/features python/features
COPY python/realtime python/realtime

ARG SDK_EXTRAS=runtime
RUN pip install --no-cache-dir "./python[${SDK_EXTRAS}]"

ENV PYTHONPATH=/app/python \
    MLAIOPS_RUNTIME_PORT=9000

EXPOSE 9000
USER nobody
ENTRYPOINT ["python", "-m", "agent_runtime"]
