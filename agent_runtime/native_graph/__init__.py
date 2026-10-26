"""Native Agent Designer graphs: the declarative interpreter shared by the
playground runner (``agent_runtime.designer``) and deployed versions
(``agents.native_graph:build_native``).

The contract lives in kiongahq/contracts ``agent-graph/``; ``components.v1.json``
here is a byte-identical copy that a test keeps in step. Graphs are data, never
code: every node kind maps to a fixed handler in ``engine.py``.
"""

from .canonical import canonical_json, content_sha256
from .engine import run_graph
from .migrate import normalize
from .validate import validate_graph

__all__ = ["canonical_json", "content_sha256", "normalize", "run_graph", "validate_graph"]
