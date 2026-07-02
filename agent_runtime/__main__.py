"""``python -m agent_runtime`` — start the runtime with uvicorn."""

from __future__ import annotations

import os

import uvicorn

from .app import create_app


def main() -> None:
    uvicorn.run(
        create_app(),
        host=os.environ.get("MLAIOPS_RUNTIME_HOST", "0.0.0.0"),
        port=int(os.environ.get("MLAIOPS_RUNTIME_PORT", "9000")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
