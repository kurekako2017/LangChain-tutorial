from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_openai import ChatOpenAI


@dataclass
class _Message:
    content: str
    reasoning_content: str = ""


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Output:
    choices: List[_Choice]


@dataclass
class _Response:
    output: _Output


class Generation:
    @staticmethod
    def call(api_key: Optional[str] = None, model: str = "", messages: Optional[List[Any]] = None, result_format: str = "message", **kwargs: Any):
        text = " ".join(str(item.get("content", "")) if isinstance(item, dict) else getattr(item, "content", str(item)) for item in (messages or []))
        response = ChatOpenAI(model=model or "qwen2.5-coder:1.5b").invoke(text)
        message = _Message(content=response.content, reasoning_content=f"Mock reasoning for {model or 'default-model'}")
        return _Response(output=_Output(choices=[_Choice(message=message)]))

