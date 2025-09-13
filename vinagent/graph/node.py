from typing import Union, Dict, Optional, Any
from langchain_core.runnables import RunnableConfig
from abc import ABC, abstractmethod


# Modified Node class with >> operator support
class Node:
    def __init__(self, name: Optional[str] = None, config: Any = None):
        self.name = name or self.__class__.__name__
        self.config = config

    @abstractmethod
    def exec(
        self, state: Optional[Any], config: Optional[RunnableConfig] = None
    ) -> Union[dict, str]:
        raise NotImplementedError("Subclasses must implement exec method")

    def branching(self, state: Any, config: Optional[RunnableConfig] = None) -> str:
        pass

    def __rshift__(self, other: Union["Node", Dict[str, "Node"], str]) -> "Node":
        self.target = other
        return self
