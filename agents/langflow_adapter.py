"""Execute a published Langflow JSON flow through the Kionga agent protocol.

The flow is mounted into the agent pod at a fixed path. LFX runs independently
of the editor workspace and emits a single complete response for each turn.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent_runtime.graphs import GraphLoadError

MAX_OUTPUT_BYTES = 4 * 1024 * 1024


def _result_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "message", "content", "result", "results", "outputs"):
            if key in result:
                text = _result_text(result[key])
                if text:
                    return text
    if isinstance(result, list):
        for item in reversed(result):
            text = _result_text(item)
            if text:
                return text
    return ""


class LangflowGraph:
    framework = "langflow"
    streaming_mode = "complete"

    def __init__(self, flow_path: Path | None = None):
        flow_path = flow_path or Path(os.environ.get("KIONGA_LANGFLOW_FLOW_PATH", "/app/flows/flow.json"))
        if not flow_path.is_file():
            raise GraphLoadError(f"published Langflow flow is missing at {flow_path}")
        self.flow_path = flow_path

    async def ainvoke(self, payload: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages") or []
        message = messages[-1]
        if not isinstance(message, HumanMessage) or not isinstance(message.content, str):
            raise ValueError("Langflow requires a text message")
        session_id = config.get("configurable", {}).get("thread_id", "")
        command = ["lfx", "run", str(self.flow_path), "--input-value", message.content,
                   "--format", "json"]
        if session_id:
            command.extend(["--session-id", str(session_id)])
        try:
            process = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=os.environ.copy(),
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=120)
        except asyncio.TimeoutError as error:
            process.kill()
            await process.communicate()
            raise RuntimeError("Langflow execution timed out") from error
        if process.returncode != 0:
            raise RuntimeError("Langflow execution failed")
        if len(stdout) > MAX_OUTPUT_BYTES:
            raise RuntimeError("Langflow result exceeds the response limit")
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as error:
            raise RuntimeError("Langflow returned invalid JSON") from error
        text = _result_text(result.get("result", result))
        if not text:
            raise RuntimeError("Langflow returned no text output")
        return {"messages": [AIMessage(content=text)]}

    async def astream(self, payload: dict[str, Any], *, config: dict[str, Any],
                      stream_mode: str = "messages"):
        result = await self.ainvoke(payload, config=config)
        yield result["messages"][-1], {}


def build_langflow() -> LangflowGraph:
    return LangflowGraph()


build_langflow.kionga_framework = "langflow"
