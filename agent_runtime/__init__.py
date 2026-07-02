"""Agent runtime: serves compiled LangGraph agents over HTTP.

This is the container entrypoint a ``NexusAgent`` runs in production and the
``agent-runtime`` Compose service locally. It loads the configured graph
module, wires the platform checkpointer, memory, and tracing, and reports
session usage back to the gateway.
"""

from .app import create_app

__all__ = ["create_app"]
