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


@agent_factory("nooa")
def build_nooa():
    if sys.version_info < (3, 12) or sys.version_info >= (3, 14):
        raise RuntimeError("The NOOA example requires Python 3.12 or 3.13")
    if os.environ.get("KIONGA_ALLOW_CODE_EXECUTION") != "1":
        raise RuntimeError("Provision an isolated sandbox before setting KIONGA_ALLOW_CODE_EXECUTION=1")
    from nooa import Agent
    from nooa.unifiedllm.registry import get_llm_client

    llm = get_llm_client(os.environ["NOOA_MODEL"])

    class Analyst(Agent, llm=llm):
        """Analyze the supplied question and return a concise explanation."""
        async def answer(self, message: str) -> str:
            """Respond to the question, making uncertainties explicit."""
            ...

    return NooaAdapter(lambda request: Analyst(), "answer")
