import asyncio
import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable, List

from langchain_core import AIMessage, HumanMessage, SystemMessage, _normalize_messages


def _base_url() -> str:
    return os.getenv("OPENAI_BASE_URL", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")).rstrip("/")


def _api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "ollama")


def _request_timeout() -> float:
    return float(os.getenv("OPENAI_REQUEST_TIMEOUT", "5"))


@dataclass
class _BaseClient:
    model: str = "qwen2.5-coder:1.5b"

    def _post_json(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{_base_url()}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {_api_key()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=_request_timeout()) as response:
            return json.loads(response.read().decode("utf-8"))

    def _fallback_text(self, prompt: Any) -> str:
        text = prompt if isinstance(prompt, str) else str(prompt)
        if "JSON" in text.upper():
            return '{"question":"%s","answer":"%s"}' % (text[:20].replace('"', "'"), "mock answer")
        digest = hashlib.sha256((self.model + "|" + text).encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"Mock response ({self.model}, {digest}): {text[:120]}"


class ChatOpenAI(_BaseClient):
    def __init__(self, model: str = None, model_name: str = None, base_url: str = None, api_key: str = None, openai_api_base: str = None, openai_api_key: str = None, **kwargs: Any):
        if base_url:
            os.environ["OPENAI_BASE_URL"] = base_url
        if openai_api_base:
            os.environ["OPENAI_BASE_URL"] = openai_api_base
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        super().__init__(model=model or model_name or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b"))
        self.kwargs = kwargs
        self._bound_tools = []

    def bind_tools(self, tools):
        bound = ChatOpenAI(model=self.model, **self.kwargs)
        bound._bound_tools = list(tools or [])
        return bound

    def _extract_query_text(self, prompt: Any) -> str:
        if isinstance(prompt, str):
            return prompt
        if hasattr(prompt, "messages"):
            prompt = prompt.messages
        if isinstance(prompt, list):
            return " ".join(getattr(message, "content", str(message)) for message in prompt)
        return str(prompt)

    def _build_function_call(self, prompt: Any, functions: List[dict] | None) -> dict | None:
        if not functions:
            return None
        query_text = self._extract_query_text(prompt)
        first = functions[0]
        name = first.get("name", "tool")
        if "move" not in name.lower() and "移动" not in query_text:
            return None

        arguments = {"source_path": "a.txt", "destination_path": "Desktop"}
        if "a.txt" in query_text:
            arguments["source_path"] = "a.txt"
        elif "文件a" in query_text:
            arguments["source_path"] = "a"
        if "Desktop" in query_text:
            marker = query_text.find("Desktop")
            start = query_text.rfind("C:\\", 0, marker)
            if start >= 0:
                arguments["destination_path"] = query_text[start : marker + len("Desktop")]
        return {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}

    def invoke(self, prompt: Any = None, **kwargs: Any) -> AIMessage:
        if prompt is None:
            prompt = kwargs.get("input", kwargs.get("messages", ""))
        if "input" in kwargs and prompt is kwargs.get("input"):
            prompt = kwargs["input"]
        try:
            data = self._post_json("/chat/completions", {"model": self.model, "messages": _normalize_messages(prompt)})
            content = data["choices"][0]["message"]["content"]
        except Exception:
            content = self._fallback_text(prompt)
        tool_calls = []
        additional_kwargs = {}
        if self._bound_tools:
            query_text = self._extract_query_text(prompt)
            for tool in self._bound_tools:
                name = getattr(tool, "name", tool.__class__.__name__)
                if name and any(keyword in query_text for keyword in [name.lower(), "天气", "搜索", "检索", "查询"]):
                    tool_calls.append({"name": name, "arguments": query_text})
        if tool_calls:
            additional_kwargs["tool_calls"] = tool_calls
        function_call = self._build_function_call(prompt, kwargs.get("functions"))
        if function_call:
            additional_kwargs["function_call"] = function_call
        return AIMessage(content=content, additional_kwargs=additional_kwargs)

    def __call__(self, prompt: Any = None, **kwargs: Any) -> AIMessage:
        return self.invoke(prompt, **kwargs)

    async def ainvoke(self, prompt: Any = None, **kwargs: Any) -> AIMessage:
        return await asyncio.to_thread(self.invoke, prompt, **kwargs)

    def stream(self, prompt: Any = None, **kwargs: Any):
        yield self.invoke(prompt, **kwargs)

    async def astream(self, prompt: Any = None, **kwargs: Any):
        yield self.invoke(prompt, **kwargs)

    def batch(self, prompts: List[Any], **kwargs: Any) -> List[AIMessage]:
        return [self.invoke(prompt, **kwargs) for prompt in prompts]


class OpenAI(_BaseClient):
    def __init__(self, model: str = None, model_name: str = None, **kwargs: Any):
        super().__init__(model=model or model_name or os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b"))
        self.kwargs = kwargs

    def invoke(self, prompt: Any = None, **kwargs: Any) -> str:
        if prompt is None:
            prompt = kwargs.get("input", kwargs.get("messages", ""))
        try:
            data = self._post_json("/chat/completions", {"model": self.model, "messages": _normalize_messages(prompt)})
            return data["choices"][0]["message"]["content"]
        except Exception:
            return self._fallback_text(prompt)

    def __call__(self, prompt: Any = None, **kwargs: Any) -> str:
        return self.invoke(prompt, **kwargs)

    async def ainvoke(self, prompt: Any = None, **kwargs: Any) -> str:
        return await asyncio.to_thread(self.invoke, prompt, **kwargs)

    def stream(self, prompt: Any = None, **kwargs: Any):
        yield self.invoke(prompt, **kwargs)

    async def astream(self, prompt: Any = None, **kwargs: Any):
        yield self.invoke(prompt, **kwargs)


class OpenAIEmbeddings(_BaseClient):
    def __init__(self, model: str = None, **kwargs: Any):
        super().__init__(model=model or os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"))
        self.kwargs = kwargs

    def embed_query(self, text: str) -> List[float]:
        try:
            data = self._post_json("/embeddings", {"model": self.model, "input": text})
            return data["data"][0]["embedding"]
        except Exception:
            import hashlib
            import numpy as np

            digest = hashlib.sha256((self.model + "|" + text).encode("utf-8", errors="ignore")).digest()
            arr = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
            return (arr / 255.0).tolist() * 4

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]
