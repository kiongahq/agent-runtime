"""Opt-in factories; provider keys come from the isolated deployment environment."""
import os
import sys

from mlaiops_sdk.agent_adapters import (
    AgentResult, AgnoAdapter, CallableAdapter, NooaAdapter, agent_factory,
)


@agent_factory("custom")
def build_custom():
    async def respond(request):
        return AgentResult(f"Hello {request.user_id or 'builder'}: {request.message}")
    return CallableAdapter(respond)


@agent_factory("agno")
def build_agno():
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    model = os.environ["AGNO_MODEL"]
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for this Agno example")
    return AgnoAdapter(lambda request: Agent(
        model=OpenAIChat(id=model), instructions="Answer clearly and identify uncertainty.",
    ))


class CodeSandboxUnavailable(RuntimeError):
    """Model-generated code cannot run because no isolated executor exists."""


def require_code_sandbox() -> None:
    """Fail closed: refuse any factory that executes model-generated code.

    NOOA runs the code it generates inside the calling process. In this
    runtime that process also serves HTTP and holds provider keys and the
    internal service token, so generated code could read those secrets and
    reach internal services. Executing it safely needs an out-of-process
    executor with no secrets, a read-only root, no network by default and
    CPU, memory and time limits. The runtime does not ship one, so this check
    always refuses; an environment flag is not a sandbox and does not
    override it. ``KIONGA_ALLOW_CODE_EXECUTION`` is no longer honored.
    """
    flag = " (KIONGA_ALLOW_CODE_EXECUTION is set but no longer enables it)" if os.environ.get("KIONGA_ALLOW_CODE_EXECUTION") else ""
    raise CodeSandboxUnavailable(
        "The NOOA example executes model-generated code in the runtime process, which holds provider keys "
        "and the internal service token. It is disabled until an isolated code-execution sandbox is "
        f"available to this runtime{flag}."
    )


@agent_factory("nooa")
def build_nooa():
    # Checked first so the refusal does not depend on the Python version or
    # on NOOA being installed.
    require_code_sandbox()
    if sys.version_info < (3, 12) or sys.version_info >= (3, 14):
        raise RuntimeError("The NOOA example requires Python 3.12 or 3.13")
    from nooa import Agent
    from nooa.unifiedllm.registry import get_llm_client

    llm = get_llm_client(os.environ["NOOA_MODEL"])

    class Analyst(Agent, llm=llm):
        """Analyze the supplied question and return a concise explanation."""
        async def answer(self, message: str) -> str:
            """Respond to the question, making uncertainties explicit."""
            ...

    return NooaAdapter(lambda request: Analyst(), "answer")
