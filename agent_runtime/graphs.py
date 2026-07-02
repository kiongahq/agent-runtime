"""Graph module loading.

The runtime is configured with ``MLAIOPS_GRAPH_MODULE`` in the same
``module.path:attribute`` form the SDK and the ``NexusAgent`` CRD use. The
attribute may be:

- a compiled LangGraph graph (used as-is),
- an uncompiled ``StateGraph`` (compiled, with a checkpointer when available),
- a callable factory ``build(model=None, checkpointer=None)`` returning either
  of the above. Factories let agents receive the platform-configured chat
  model and checkpointer without importing platform config themselves.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any


class GraphLoadError(RuntimeError):
    pass


def resolve_attribute(spec: str) -> Any:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise GraphLoadError(
            f"graph module {spec!r} must use the form 'package.module:attribute'"
        )
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise GraphLoadError(f"cannot import graph module {module_name!r}: {error}") from error
    try:
        return getattr(module, attribute)
    except AttributeError:
        raise GraphLoadError(f"module {module_name!r} has no attribute {attribute!r}") from None


def build_graph(spec: str, *, model: Any = None, checkpointer: Any = None) -> Any:
    """Load ``spec`` and return a compiled, invokable graph."""
    target = resolve_attribute(spec)
    if callable(target) and not hasattr(target, "invoke"):
        kwargs: dict[str, Any] = {}
        parameters = inspect.signature(target).parameters
        if "model" in parameters:
            kwargs["model"] = model
        if "checkpointer" in parameters:
            kwargs["checkpointer"] = checkpointer
        target = target(**kwargs)
    compile_method = getattr(target, "compile", None)
    if compile_method is not None and not hasattr(target, "invoke"):
        target = compile_method(checkpointer=checkpointer)
    if not hasattr(target, "ainvoke"):
        raise GraphLoadError(
            f"graph module {spec!r} did not produce an invokable graph "
            "(expected a compiled LangGraph graph, a StateGraph, or a factory)"
        )
    return target
