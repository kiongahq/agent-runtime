# Agent runtime image: serves any platform-registered LangGraph agent.
# Build from the repository root:
#   docker build -f python/agent_runtime/Dockerfile -t mlaiops/agent-runtime .
FROM python:3.11-slim

WORKDIR /app

COPY python/pyproject.toml python/pyproject.toml
COPY python/mlaiops_sdk python/mlaiops_sdk
COPY python/agent_runtime python/agent_runtime
COPY python/agents python/agents
COPY python/features python/features
COPY python/realtime python/realtime

RUN pip install --no-cache-dir "./python[runtime]"

ENV PYTHONPATH=/app/python \
    MLAIOPS_RUNTIME_PORT=9000

EXPOSE 9000
USER nobody
ENTRYPOINT ["python", "-m", "agent_runtime"]
