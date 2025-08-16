import sys
from types import MappingProxyType
from typing import Any, Literal, Mapping, cast
from typing import Optional, Any, Union, Literal
from langchain_core.runnables import RunnableConfig
from vinagent.graph.node import Node


class SpecialNonExecNode(Node):
    def __init__(self, name: Literal["__start__", "__end__"]):
        super().__init__(name=name)  # Match LangGraph's START constant name

    def exec(
        self, state: Any, config: Optional[RunnableConfig] = None
    ) -> Union[dict, str]:
        raise RuntimeError("StartNode is a control node and should not be executed")

    @property
    def is_branching(self) -> bool:
        return False

    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return super().__eq__(other)

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return self.name


# Create a single instance of StartNode to use as the START constant
START = SpecialNonExecNode(name="__start__")
END = SpecialNonExecNode(name="__end__")


# --- Empty read-only containers ---
EMPTY_MAP: Mapping[str, Any] = MappingProxyType({})
EMPTY_SEQ: tuple[str, ...] = tuple()
MISSING = object()
NS_SEP = sys.intern("|")
# for checkpoint_ns, separates each level (ie. graph|subgraph|subsubgraph)
NS_END = sys.intern(":")

"""Tag to disable streaming for a chat model."""
TAG_HIDDEN = sys.intern("langsmith:hidden")
# marker to signal node was scheduled (in distributed mode)
TASKS = sys.intern("__pregel_tasks")

INTERRUPT = sys.intern("__interrupt__")
