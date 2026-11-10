"""Deterministic model for ``POST /agent-turns`` under ``KIONGA_LLM_BACKEND=mock``.

It is a real LangChain chat model, so a mock turn goes through the same message
conversion, ``bind_tools`` and reply extraction as a provider turn. The decision is a
pure function of the transcript and the bound tools.

**Default policy.** A turn with a ``spawn_worker`` tool is a coordinator: it spawns each
member listed under ``Team members you can spawn:`` in the system prompt once, replies with
text and no tool calls while it waits, and calls ``finish`` when every spawned member has
reported back in a ``<kionga-data source="worker:NAME" kind="hand-back">`` frame, which is
also how a failed worker's outcome must arrive. Any other turn is a worker: it runs one safe
``exec`` when granted, then calls ``finish``.

**Scripts.** The first user message may hold a JSON list of steps between two
``[[mock-script]]`` markers. This is read only here, so it has no effect on any real
backend, and only from the first user message, so tool output cannot supply one::

    [[mock-script]][
      {"tool_calls": [{"name": "exec", "arguments": {"command": "true"}}]},
      {"error": {"code": "rate_limited", "retry_after_seconds": 1}, "times": 2},
      {"content": "text", "usage": {"input_tokens": 900, "output_tokens": 100}}
    ][[mock-script]]

A step's position is the number of assistant messages so far, so the runner stays
stateless, with one test-only exception: ``error`` steps add no assistant message, so
each identical transcript's attempts are counted in this process (``times`` defaults to
1). When the steps run out, the default policy applies. A script may call a tool that was
not offered, which is how a test exercises the gateway's refusal. Nest a worker's script
in a ``spawn_worker`` task string; the markers match greedily, so the outer script keeps
the inner one intact.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from .agent_turns import ERROR_CODES, TurnError

MARKER = "[[mock-script]]"
SCRIPT = re.compile(re.escape(MARKER) + r"(.*)" + re.escape(MARKER), re.DOTALL)
FRAME = re.compile(r'<kionga-data\s+source="worker:([^"]+)"\s+kind="([^"]+)"')
MEMBER_HEADING = "Team members you can spawn:"
MEMBER_LINE = re.compile(r"^- ([a-z][a-z0-9-]{0,62}):", re.MULTILINE)
SAFE_COMMAND = "echo kionga-mock-worker"
STEP_KEYS = {"content", "tool_calls", "usage", "error", "times"}
MAX_ATTEMPTS_TRACKED = 4096


def report_text(result: str, evidence: str) -> str:
    return (f"Result: {result}\nEvidence: {evidence}\n"
            "Not verified: this is the deterministic mock backend, so no work was independently verified.\n"
            "Follow-ups: none.")


def _invalid(reason: str) -> TurnError:
    return TurnError("invalid_request", f"invalid {MARKER}: {reason}")


def _check_step(step: Any) -> None:
    if not isinstance(step, dict) or not step or set(step) - STEP_KEYS:
        raise _invalid(f"each step is an object with keys from {sorted(STEP_KEYS)}")
    if "error" in step:
        error, times = step["error"], step.get("times", 1)
        if not isinstance(error, dict) or error.get("code") not in ERROR_CODES:
            raise _invalid(f"error.code must be one of {list(ERROR_CODES)}")
        if not isinstance(times, int) or isinstance(times, bool) or times < 1:
            raise _invalid("times must be a positive integer")
        return
    if not isinstance(step.get("content", ""), str):
        raise _invalid("content must be a string")
    calls = step.get("tool_calls", [])
    if not isinstance(calls, list) or not all(
            isinstance(call, dict) and isinstance(call.get("name"), str) and isinstance(call.get("arguments", {}), dict)
            for call in calls):
        raise _invalid("tool_calls must be a list of {name, arguments}")


def parse_script(messages: list[Any]) -> list[dict[str, Any]] | None:
    first = next((message for message in messages if isinstance(message, HumanMessage)), None)
    text = str(first.content) if first is not None else ""
    if MARKER not in text:
        return None
    match = SCRIPT.search(text)
    if match is None:
        raise _invalid("needs an opening and a closing marker")
    try:
        script = json.loads(match.group(1))
    except ValueError:
        raise _invalid("the script is not valid JSON") from None
    if not isinstance(script, list):
        raise _invalid("the script must be a JSON list of steps")
    for step in script:
        _check_step(step)
    return script


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    return {tool["function"]["name"] for tool in tools}


def _members(messages: list[Any]) -> list[str]:
    """Member names from the gateway's assembled system prompt."""
    system = next((str(message.content) for message in messages if isinstance(message, SystemMessage)), "")
    _, found, listing = system.partition(MEMBER_HEADING)
    return MEMBER_LINE.findall(listing.split("\n\n", 1)[0]) if found else []


def _finish_or_text(names: set[str], messages: list[Any], report: str) -> dict[str, Any]:
    if "finish" not in names:
        return {"content": report}
    last_reply = next((message for message in reversed(messages) if isinstance(message, AIMessage)), None)
    if (isinstance(messages[-1], ToolMessage) and last_reply is not None
            and any(call["name"] == "finish" for call in last_reply.tool_calls)):
        # A refused finish: wait for the next delivery instead of retrying in a loop.
        return {"content": "Waiting."}
    return {"tool_calls": [{"name": "finish", "arguments": {"report": report}}]}


def _coordinator_step(messages: list[Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    results = {message.tool_call_id: str(message.content) for message in messages if isinstance(message, ToolMessage)}
    spawned: dict[str, bool] = {}
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] == "spawn_worker":
                    agent = str(call["args"].get("agent", ""))
                    spawned[agent] = spawned.get(agent, False) or '"error"' not in results.get(call["id"], "")
    missing = [member for member in _members(messages) if member not in spawned]
    if missing:
        return {"tool_calls": [
            {"name": "spawn_worker", "arguments": {"agent": member, "task": f"Handle your part of the request as {member}."}}
            for member in missing
        ]}
    reported = {source for message in messages if isinstance(message, HumanMessage)
                for source, kind in FRAME.findall(str(message.content)) if kind == "hand-back"}
    live = sorted(agent for agent, started in spawned.items() if started and agent not in reported)
    if live:
        return {"content": f"Waiting for: {', '.join(live)}."}
    finished = sorted(agent for agent, started in spawned.items() if started)
    if not finished:
        result = "no members were spawned; the system prompt listed none or every spawn failed"
    else:
        result = "; ".join(f"{agent}: reported back" for agent in finished)
    return _finish_or_text(_tool_names(tools), messages, report_text(result, f"{len(finished)} worker report(s) received."))


def _worker_step(messages: list[Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    names = _tool_names(tools)
    ran_exec = any(call["name"] == "exec" for message in messages if isinstance(message, AIMessage)
                   for call in message.tool_calls)
    if "exec" in names and not ran_exec:
        return {"tool_calls": [{"name": "exec", "arguments": {"command": SAFE_COMMAND}}]}
    evidence = "ran one exec." if ran_exec else "no exec tool was granted."
    return _finish_or_text(names, messages, report_text("the mock task is complete", evidence))


def _default_step(messages: list[Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    if "spawn_worker" in _tool_names(tools):
        return _coordinator_step(messages, tools)
    return _worker_step(messages, tools)


class ScriptedTurnModel(BaseChatModel):
    _attempts: OrderedDict[str, int] = PrivateAttr(default_factory=OrderedDict)
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "kionga-agent-turn-mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.bind(tools=list(tools), **kwargs)

    def _attempt(self, messages: list[Any], index: int) -> int:
        digest = hashlib.sha256(repr((index, [(type(message).__name__, message.content, getattr(message, "tool_calls", None))
                                              for message in messages])).encode()).hexdigest()
        with self._lock:
            count = self._attempts.pop(digest, 0) + 1
            self._attempts[digest] = count
            while len(self._attempts) > MAX_ATTEMPTS_TRACKED:
                self._attempts.popitem(last=False)
        return count

    def _scripted_step(self, script: list[dict[str, Any]], messages: list[Any]) -> dict[str, Any] | None:
        remaining = sum(isinstance(message, AIMessage) for message in messages)
        for index, step in enumerate(script):
            if "error" in step:
                if remaining == 0 and self._attempt(messages, index) <= step.get("times", 1):
                    error = step["error"]
                    raise TurnError(error["code"], str(error.get("message", f"scripted {error['code']}")),
                                    error.get("retry_after_seconds"))
            elif remaining == 0:
                return step
            else:
                remaining -= 1
        return None

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        tools = kwargs.get("tools") or []
        script = parse_script(messages)
        step = self._scripted_step(script, messages) if script is not None else None
        if step is None:
            step = _default_step(messages, tools)
        number = sum(isinstance(message, AIMessage) for message in messages)
        calls = [{"id": call.get("id") or f"mock-{number}-{index}", "name": call["name"],
                  "args": call.get("arguments", {}), "type": "tool_call"}
                 for index, call in enumerate(step.get("tool_calls", []))]
        content = step.get("content", "")
        usage = step.get("usage") or {}
        input_tokens = usage.get("input_tokens", sum(len(str(message.content).split()) for message in messages))
        output_tokens = usage.get("output_tokens", len(content.split()) + len(json.dumps(calls).split()) * bool(calls))
        message = AIMessage(content=content, tool_calls=calls, response_metadata={"model_name": "mock"}, usage_metadata={
            "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens,
        })
        return ChatResult(generations=[ChatGeneration(message=message)])
