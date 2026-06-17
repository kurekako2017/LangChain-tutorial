from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class BaseMessage:
    content: str
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanMessage(BaseMessage):
    pass


@dataclass
class SystemMessage(BaseMessage):
    pass


@dataclass
class AIMessage(BaseMessage):
    pass

