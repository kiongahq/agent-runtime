# syntax=docker/dockerfile:1
# Agent runtime image: serves a LangGraph agent or a marked Agno/NOOA/custom
# adapter over HTTP. The same image runs the feature materializer and the
# realtime processor (different entrypoints).
#
#   docker build -t ghcr.io/kiongahq/agent-runtime .                  # SDK from its release
#   docker build --build-arg SDK_SOURCE=local \
#                --build-context sdk=../sdk-python -t ... .            # SDK from a sibling checkout
#
# SDK_SOURCE is explicit (not inferred from the context) so it is part of the
# layer cache key: a release build can never reuse a layer built from local code.
ARG PYTHON_VERSION=3.11

# Empty unless overridden with --build-context sdk=<dir>.
FROM scratch AS sdk

FROM python:${PYTHON_VERSION}-slim
WORKDIR /app

ARG SDK_SOURCE=release
ARG SDK_VERSION=v0.2.0
ARG SDK_EXTRAS=runtime
COPY --from=sdk / /opt/sdk/
RUN case "$SDK_SOURCE" in \
      local) test -f /opt/sdk/pyproject.toml \
               || { echo "SDK_SOURCE=local needs --build-context sdk=<sdk-python checkout>" >&2; exit 1; }; \
             pip install --no-cache-dir "/opt/sdk[${SDK_EXTRAS}]" ;; \
      release) pip install --no-cache-dir \
               "mlaiops-sdk[${SDK_EXTRAS}] @ https://github.com/kiongahq/sdk-python/archive/refs/tags/${SDK_VERSION}.tar.gz" ;; \
      *) echo "SDK_SOURCE must be release or local, got: $SDK_SOURCE" >&2; exit 1 ;; \
    esac && rm -rf /opt/sdk

# Publish a separate Langflow runtime flavor. Native agent images keep the
# smaller dependency set; exported 1.13.x flows run with matching LFX 1.13.x.
ARG RUNTIME_FLAVOR=standard
ARG LFX_VERSION=1.13.0
RUN case "$RUNTIME_FLAVOR" in \
      standard) ;; \
      langflow) pip install --no-cache-dir "lfx[bundles]==${LFX_VERSION}" ;; \
      *) echo "Unknown runtime flavor: $RUNTIME_FLAVOR" >&2; exit 1 ;; \
    esac

COPY agent_runtime /app/agent_runtime
COPY agents /app/agents
COPY features /app/features
COPY realtime /app/realtime

ENV PYTHONPATH=/app \
    LANGFLOW_COMPONENTS_PATH=/app/agent_runtime/langflow_components \
    MLAIOPS_RUNTIME_PORT=9000

EXPOSE 9000
USER nobody
ENTRYPOINT ["python", "-m", "agent_runtime"]
