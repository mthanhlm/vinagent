from typing import List
from langchain_core.messages import BaseMessage
from collections import deque


class InConversationHistory:
    def __init__(self, messages: List[BaseMessage] = [], max_length: int = 10):
        self.max_length = max_length
        self.history = deque(iterable=messages, maxlen=max_length)

    def add_message(self, message: BaseMessage) -> None:
        self.history.append(message)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.history.extend(messages)

    def pop_left(self) -> None:
        self.history.popleft()

    def pop(self) -> None:
        self.history.pop()

    def append(self) -> None:
        self.history.append()

    def append_left(self) -> None:
        self.history.appendleft()

    def get_history(self, max_history: int = None) -> List[BaseMessage]:
        len_history = len(self.history)
        if max_history:
            return list(self.history)[-min(max_history, len_history) :]
        else:
            return list(self.history)
