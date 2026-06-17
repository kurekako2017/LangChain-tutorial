import csv
import json
import sys
import types
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core import Document, FAISS, Chroma, ChatMessageHistory, RecursiveCharacterTextSplitter, StructuredTool, Tool


def _register(name: str, attrs: Dict[str, Any], is_package: bool = False):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    if is_package:
        module.__path__ = []
    sys.modules[name] = module
    return module


class TextLoader:
    def __init__(self, file_path: str, encoding: Optional[str] = None, **kwargs: Any):
        self.file_path = file_path
        self.encoding = encoding
        self.kwargs = kwargs

    def load(self):
        encodings = [self.encoding] if self.encoding else ["utf-8", "gbk", "utf-8-sig", "latin-1"]
        for enc in encodings:
            if not enc:
                continue
            try:
                return [Document(Path(self.file_path).read_text(encoding=enc), {"source": self.file_path})]
            except Exception:
                continue
        return [Document(Path(self.file_path).read_bytes().decode("utf-8", errors="ignore"), {"source": self.file_path})]

    def load_and_split(self):
        return self.load()


class CSVLoader:
    def __init__(self, file_path: str, encoding: Optional[str] = None, source_column: Optional[str] = None, **kwargs: Any):
        self.file_path = file_path
        self.encoding = encoding
        self.source_column = source_column
        self.kwargs = kwargs

    def load(self):
        out = []
        with open(self.file_path, "r", encoding=self.encoding or "utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                metadata = {"row": i, "source": self.file_path}
                if self.source_column and self.source_column in row:
                    metadata["source_value"] = row[self.source_column]
                out.append(Document(json.dumps(row, ensure_ascii=False), metadata))
        return out

    def load_and_split(self):
        return self.load()


class JSONLoader:
    def __init__(self, file_path: str, jq_schema: str = ".", text_content: bool = True, **kwargs: Any):
        self.file_path = file_path
        self.jq_schema = jq_schema
        self.text_content = text_content
        self.kwargs = kwargs

    def load(self):
        data = json.loads(Path(self.file_path).read_text(encoding="utf-8", errors="ignore"))
        return [Document(json.dumps(data, ensure_ascii=False), {"source": self.file_path})]

    def load_and_split(self):
        return self.load()


class PyPDFLoader:
    def __init__(self, file_path: str, **kwargs: Any):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        try:
            from pypdf import PdfReader

            reader = PdfReader(self.file_path)
            return [Document(page.extract_text() or "", {"page": i, "source": self.file_path}) for i, page in enumerate(reader.pages)]
        except Exception:
            return [Document(f"PDF loader unavailable for {self.file_path}", {"source": self.file_path})]

    def load_and_split(self):
        return self.load()


class WebBaseLoader:
    def __init__(self, web_path: str, bs_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self.web_path = web_path
        self.bs_kwargs = bs_kwargs or {}
        self.kwargs = kwargs

    def load(self):
        try:
            with urllib.request.urlopen(self.web_path, timeout=20) as response:
                text = response.read().decode("utf-8", errors="ignore")
            if self.bs_kwargs:
                from bs4 import BeautifulSoup

                parse_only = self.bs_kwargs.get("parse_only")
                soup = BeautifulSoup(text, "html.parser", parse_only=parse_only) if parse_only is not None else BeautifulSoup(text, "html.parser")
                text = soup.get_text("\n", strip=True)
            return [Document(text, {"source": self.web_path})]
        except Exception:
            return [Document(f"Unable to fetch {self.web_path}", {"source": self.web_path})]

    def load_and_split(self):
        return self.load()


class UnstructuredHTMLLoader:
    def __init__(self, file_path: str, **kwargs: Any):
        self.file_path = file_path
        self.kwargs = kwargs

    def load(self):
        text = Path(self.file_path).read_text(encoding="utf-8", errors="ignore")
        return [Document(text, {"source": self.file_path})]


class UnstructuredMarkdownLoader(UnstructuredHTMLLoader):
    pass


class PythonLoader(TextLoader):
    pass


class DirectoryLoader:
    def __init__(self, path: str, glob: str = "**/*", loader_cls: Any = TextLoader, loader_kwargs: Optional[Dict[str, Any]] = None, **kwargs: Any):
        self.path = Path(path)
        self.glob = glob
        self.loader_cls = loader_cls
        self.loader_kwargs = loader_kwargs or {}
        self.kwargs = kwargs

    def load(self):
        docs = []
        for file_path in self.path.glob(self.glob):
            if file_path.is_file():
                loader = self.loader_cls(str(file_path), **self.loader_kwargs)
                docs.extend(loader.load())
        return docs


class SQLDatabase:
    from langchain_core import SQLDatabase as _SQLDatabase

    @classmethod
    def from_uri(cls, uri: str):
        return cls._SQLDatabase.from_uri(uri)


class TavilySearchResults(StructuredTool):
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        super().__init__("TavilySearchResults", "local search stub", self._search)

    def _search(self, query: str):
        return f"Local Tavily stub: {query}"


class MoveFileTool(StructuredTool):
    def __init__(self):
        super().__init__("MoveFileTool", "move file", self._move)

    def _move(self, source_path: str = "", destination_path: str = ""):
        source = Path(source_path)
        destination = Path(destination_path)
        if not source.exists():
            return f"{source_path} does not exist"
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        return f"Moved {source_path} to {destination_path}"


_register(
    __name__ + ".document_loaders",
    {
        "TextLoader": TextLoader,
        "CSVLoader": CSVLoader,
        "JSONLoader": JSONLoader,
        "PyPDFLoader": PyPDFLoader,
        "WebBaseLoader": WebBaseLoader,
        "UnstructuredHTMLLoader": UnstructuredHTMLLoader,
        "UnstructuredMarkdownLoader": UnstructuredMarkdownLoader,
        "DirectoryLoader": DirectoryLoader,
        "PythonLoader": PythonLoader,
    },
    is_package=True,
)
_register(__name__ + ".document_loaders.pdf", {"PyPDFLoader": PyPDFLoader})
_register(__name__ + ".vectorstores", {"FAISS": FAISS, "Chroma": Chroma})
_register(__name__ + ".chat_message_histories", {"ChatMessageHistory": ChatMessageHistory})
_register(__name__ + ".utilities", {"SQLDatabase": SQLDatabase}, is_package=True)
_register(__name__ + ".utilities.sql_database", {"SQLDatabase": SQLDatabase})
_register(__name__ + ".tools", {"MoveFileTool": MoveFileTool}, is_package=True)
_register(__name__ + ".tools.tavily_search", {"TavilySearchResults": TavilySearchResults})
