from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


class ChatOllama(ChatOpenAI):
    def __init__(self, model: str = None, **kwargs: Any):
        super().__init__(model=model or "qwen2.5-coder:1.5b", **kwargs)


class OllamaEmbeddings(OpenAIEmbeddings):
    def __init__(self, model: str = None, **kwargs: Any):
        super().__init__(model=model or "nomic-embed-text", **kwargs)

