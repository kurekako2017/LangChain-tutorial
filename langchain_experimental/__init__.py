import sys
import types
from typing import Any, Dict

from langchain_core import Document, PythonREPL


def _register(name: str, attrs: Dict[str, Any], is_package: bool = False):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    if is_package:
        module.__path__ = []
    sys.modules[name] = module
    return module


_register(__name__ + ".utilities", {}, is_package=True)
_register(__name__ + ".utilities.python", {"PythonREPL": PythonREPL})


class SemanticChunker:
    def __init__(self, embeddings: Any = None, chunk_size: int = 1000, **kwargs: Any):
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.kwargs = kwargs

    def split_documents(self, docs):
        out = []
        for doc in docs:
            text = doc.page_content
            for start in range(0, len(text), self.chunk_size):
                out.append(Document(text[start : start + self.chunk_size], dict(doc.metadata)))
        return out

    def create_documents(self, texts):
        return [Document(text, {}) for text in texts]


_register(__name__ + ".text_splitter", {"SemanticChunker": SemanticChunker})
