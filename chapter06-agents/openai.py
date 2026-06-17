from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_openai import ChatOpenAI

__version__ = "1.81.0-local"


@dataclass
class _CompletionChoice:
    text: str = ""
    message: Any = None


@dataclass
class _CompletionResponse:
    choices: List[_CompletionChoice]


class _Completions:
    def __init__(self, client: "OpenAI"):
        self.client = client

    def create(self, model: str, prompt: Optional[str] = None, messages: Optional[List[Any]] = None, **kwargs: Any):
        if messages is not None:
            text = " ".join(str(m.get("content", "")) if isinstance(m, dict) else getattr(m, "content", str(m)) for m in messages)
            response = ChatOpenAI(model=model).invoke(text)
            return _CompletionResponse([_CompletionChoice(text=response.content)])
        text = prompt or ""
        response = ChatOpenAI(model=model).invoke(text)
        return _CompletionResponse([_CompletionChoice(text=response.content)])


class _ChatMessage:
    def __init__(self, content: str, reasoning_content: str = ""):
        self.content = content
        self.reasoning_content = reasoning_content


class _ChatCompletions:
    def __init__(self, client: "OpenAI"):
        self.client = client

    def create(self, model: str, messages: List[Any], **kwargs: Any):
        text = " ".join(str(m.get("content", "")) if isinstance(m, dict) else getattr(m, "content", str(m)) for m in messages)
        response = ChatOpenAI(model=model).invoke(text)
        message = _ChatMessage(content=response.content, reasoning_content=f"Mock reasoning for {model}")
        return _CompletionResponse([_CompletionChoice(message=message)])


class _Chat:
    def __init__(self, client: "OpenAI"):
        self.completions = _ChatCompletions(client)


class OpenAI:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs: Any):
        self.api_key = api_key
        self.base_url = base_url
        self.kwargs = kwargs
        self.completions = _Completions(self)
        self.chat = _Chat(self)
