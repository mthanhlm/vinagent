import inspect
from abc import ABC, abstractmethod
from typing import Union, Dict, Optional, Any
from langchain_core.runnables import RunnableConfig
from vinagent.graph.node import Node
from vinagent.agent import Agent


class AgentNode(Agent, Node):
    def __init__(self, name: Optional[str] = None, config: Any = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name or self.__class__.__name__
        self.config = config

    def branching(self, state: Any, config: Optional[RunnableConfig] = None) -> str:
        pass

    def __rshift__(self, other: Union["Node", Dict[str, "Node"], str]) -> "Node":
        self.target = other
        return self


class UserFeedback(ABC, Node):
    def __init__(
        self, name: str = "user_feedback", role: str = "user", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        self.role = role

    @abstractmethod
    def exec(
        self, state: Optional[Any], config: Optional[RunnableConfig] = None
    ) -> Union[dict, str]:
        raise NotImplementedError("Subclasses must implement exec method")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        child_sig = inspect.signature(cls.exec)

        # Compare parameter details
        child_params = [p.name for p in child_sig.parameters.values()]

        if "state" not in child_params:
            print(child_params)
            raise TypeError(
                f"Your are missing 'state' argument in exec() method. Fix by adding like: exec(self, state: State)"
            )
