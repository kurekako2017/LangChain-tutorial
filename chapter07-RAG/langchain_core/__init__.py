import csv
import json
import os
import re
import sqlite3
import sys
import types
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - optional dependency in local shim
    np = None


def _register(name: str, attrs: Dict[str, Any], is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    if is_package:
        module.__path__ = []
    sys.modules[name] = module
    return module


def _as_text(obj: Any) -> str:
    if hasattr(obj, "content"):
        return getattr(obj, "content")
    if isinstance(obj, dict) and "content" in obj:
        return str(obj["content"])
    return str(obj)


def _normalize_messages(messages: Any) -> List[Dict[str, str]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if hasattr(messages, "messages"):
        messages = getattr(messages, "messages")
    out = []
    for message in messages:
        if isinstance(message, dict):
            out.append({"role": message.get("role", "user"), "content": str(message.get("content", ""))})
        else:
            role = "user"
            cls = message.__class__.__name__.lower()
            if "system" in cls:
                role = "system"
            elif "ai" in cls or "assistant" in cls:
                role = "assistant"
            out.append({"role": role, "content": getattr(message, "content", str(message))})
    return out


@dataclass
class BaseMessage:
    content: str
    additional_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage(BaseMessage):
    role: str = "assistant"


class HumanMessage(BaseMessage):
    pass


class SystemMessage(BaseMessage):
    pass


class AIMessage(BaseMessage):
    @property
    def tool_calls(self):
        return self.additional_kwargs.get("tool_calls", [])

    pass


@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class Runnable:
    def __or__(self, other: Any):
        return RunnableSequence([self, other])

    def invoke(self, input: Any):
        raise NotImplementedError


class RunnableSequence(Runnable):
    def __init__(self, steps: List[Any]):
        self.steps = []
        for step in steps:
            if isinstance(step, RunnableSequence):
                self.steps.extend(step.steps)
            else:
                self.steps.append(step)

    def __or__(self, other: Any):
        return RunnableSequence(self.steps + ([other] if not isinstance(other, RunnableSequence) else other.steps))

    def invoke(self, input: Any):
        value = input
        for step in self.steps:
            value = step.invoke(value) if hasattr(step, "invoke") else step(value)
        return value


class StringPromptValue(str):
    @property
    def text(self):
        return str(self)

    def to_string(self):
        return str(self)


class PromptTemplate(Runnable):
    def __init__(self, template: str, input_variables: Optional[List[str]] = None, partial_variables: Optional[Dict[str, Any]] = None):
        self.template = template
        self.input_variables = input_variables or []
        self.partial_variables = partial_variables or {}

    @classmethod
    def from_template(cls, template: str, **kwargs: Any):
        return cls(template, **kwargs)

    def partial(self, **kwargs: Any):
        partial_variables = dict(self.partial_variables)
        partial_variables.update(kwargs)
        return PromptTemplate(self.template, self.input_variables, partial_variables)

    def __add__(self, other: Any):
        return PromptTemplate(self.template + str(other), self.input_variables, dict(self.partial_variables))

    def format(self, **kwargs: Any) -> str:
        payload = dict(self.partial_variables)
        payload.update(kwargs)
        return self.template.format(**payload)

    def invoke(self, input: Any):
        if isinstance(input, dict):
            return StringPromptValue(self.format(**input))
        return StringPromptValue(self.format(input=input))


class MessagesPlaceholder:
    def __init__(self, variable_name: str = "messages"):
        self.variable_name = variable_name


class HumanMessagePromptTemplate:
    @classmethod
    def from_template(cls, template: str):
        return ("human", template)


class SystemMessagePromptTemplate:
    @classmethod
    def from_template(cls, template: str):
        return ("system", template)


class ChatPromptTemplate(Runnable):
    def __init__(self, messages: Optional[List[Any]] = None, input_variables: Optional[List[str]] = None, **kwargs: Any):
        if messages is None and "messages" in kwargs:
            messages = kwargs.pop("messages")
        self.messages = messages or []
        self.input_variables = input_variables or kwargs.pop("input_variables", []) or []
        self.kwargs = kwargs

    @classmethod
    def from_messages(cls, messages: List[Any]):
        return cls(messages)

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        out: List[BaseMessage] = []
        for item in self.messages:
            if isinstance(item, MessagesPlaceholder):
                out.extend(kwargs.get(item.variable_name, []))
                continue
            if isinstance(item, BaseMessage):
                out.append(item)
                continue
            if hasattr(item, "format_messages"):
                out.extend(item.format_messages(**kwargs))
                continue
            if isinstance(item, str):
                out.append(HumanMessage(item.format(**kwargs)))
                continue
            role, template = item
            text = template.format(**kwargs)
            if role == "system":
                out.append(SystemMessage(text))
            elif role == "ai":
                out.append(AIMessage(text))
            else:
                out.append(HumanMessage(text))
        return out

    def format(self, **kwargs: Any) -> str:
        return "\n".join(message.content for message in self.format_messages(**kwargs))

    def format_prompt(self, **kwargs: Any):
        return ChatPromptValue(self.format_messages(**kwargs))

    def invoke(self, input: Any):
        if isinstance(input, dict):
            return ChatPromptValue(self.format_messages(**input))
        return ChatPromptValue(self.format_messages(input=input))


class ChatPromptValue:
    def __init__(self, messages: List[BaseMessage]):
        self.messages = messages

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def to_messages(self):
        return list(self.messages)

    def to_string(self):
        return "\n".join(message.content for message in self.messages)


class FewShotPromptTemplate(PromptTemplate):
    def __init__(self, examples: Optional[List[Dict[str, Any]]] = None, example_prompt: PromptTemplate = None, prefix: str = "", suffix: str = "", input_variables: Optional[List[str]] = None, example_selector: Any = None, **kwargs: Any):
        self.examples = examples or []
        self.example_selector = example_selector
        self.example_prompt = example_prompt
        self.prefix = prefix
        self.suffix = suffix
        super().__init__("", input_variables)

    def format(self, **kwargs: Any) -> str:
        body = [self.prefix] if self.prefix else []
        examples = self.example_selector.select_examples(kwargs) if self.example_selector is not None else self.examples
        for example in examples:
            body.append(self.example_prompt.format(**example))
        if self.suffix:
            body.append(self.suffix.format(**kwargs))
        return "\n".join(part for part in body if part)


class FewShotChatMessagePromptTemplate:
    def __init__(self, examples: List[Dict[str, Any]], example_prompt: ChatPromptTemplate):
        self.examples = examples
        self.example_prompt = example_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        messages: List[BaseMessage] = []
        for example in self.examples:
            messages.extend(self.example_prompt.format_messages(**example))
        return messages

    def format(self, **kwargs: Any) -> str:
        return "\n".join(message.content for message in self.format_messages(**kwargs))


def load_prompt(path: str, encoding: str = "utf-8", **_: Any):
    text = Path(path).read_text(encoding=encoding)
    if path.endswith((".json", ".yaml", ".yml")):
        try:
            data = json.loads(text)
        except Exception:
            data = None
            if path.endswith((".yaml", ".yml")):
                try:
                    import yaml

                    data = yaml.safe_load(text)
                except Exception:
                    data = None
        if isinstance(data, dict) and "template" in data:
            template = data["template"]
            if isinstance(template, (list, tuple)):
                template = "\n".join(str(item) for item in template)
            return PromptTemplate.from_template(str(template))
    return PromptTemplate.from_template(text)


class StrOutputParser(Runnable):
    def invoke(self, input: Any):
        return _as_text(input)

    def parse(self, text: Any):
        return _as_text(text)


class JsonOutputParser(Runnable):
    def invoke(self, input: Any):
        text = _as_text(input).strip()
        try:
            return json.loads(text)
        except Exception:
            match = re.search(r"\{.*\}", text, re.S)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            return {"text": text}

    def parse(self, text: Any):
        return self.invoke(text)

    def get_format_instructions(self):
        return 'Return a valid JSON object such as {"question": "...", "answer": "..."}.'


class XMLOutputParser(Runnable):
    def invoke(self, input: Any):
        import xml.etree.ElementTree as ET

        text = _as_text(input)
        try:
            return ET.fromstring(text)
        except Exception:
            root = ET.Element("text")
            root.text = text
            return root

    def parse(self, text: Any):
        return self.invoke(text)

    def get_format_instructions(self):
        return "Return a valid XML document."


class CommaSeparatedListOutputParser(Runnable):
    def invoke(self, input: Any):
        return [part.strip() for part in _as_text(input).split(",") if part.strip()]

    def get_format_instructions(self):
        return "Return a comma-separated list."

    def parse(self, text: Any):
        return self.invoke(text)


class DatetimeOutputParser(Runnable):
    def invoke(self, input: Any):
        from datetime import datetime

        text = _as_text(input).strip()
        for candidate in [text, text.replace("Z", "+00:00")]:
            try:
                return datetime.fromisoformat(candidate)
            except Exception:
                continue
        return text

    def parse(self, text: Any):
        return self.invoke(text)

    def get_format_instructions(self):
        return "Return a datetime in ISO 8601 format, for example 2026-06-17T00:00:00."


class BaseChatMessageHistory:
    def __init__(self):
        self.messages: List[BaseMessage] = []

    def add_user_message(self, content: str):
        self.messages.append(HumanMessage(content))

    def add_ai_message(self, content: str):
        self.messages.append(AIMessage(content))

    def clear(self):
        self.messages.clear()


class ChatMessageHistory(BaseChatMessageHistory):
    pass


class ConversationBufferMemory:
    def __init__(self, memory_key: str = "history", return_messages: bool = False, chat_memory: Optional[BaseChatMessageHistory] = None, **kwargs: Any):
        self.memory_key = memory_key
        self.return_messages = return_messages
        self.chat_memory = chat_memory or ChatMessageHistory()
        self.kwargs = kwargs

    def load_memory_variables(self, _: Dict[str, Any]):
        if self.return_messages:
            return {self.memory_key: list(self.chat_memory.messages)}
        return {self.memory_key: "\n".join(_as_text(m) for m in self.chat_memory.messages)}

    @property
    def buffer(self):
        if self.return_messages:
            return list(self.chat_memory.messages)
        return "\n".join(_as_text(m) for m in self.chat_memory.messages)

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        self.chat_memory.add_user_message(_as_text(inputs))
        self.chat_memory.add_ai_message(_as_text(outputs))


class ConversationBufferWindowMemory(ConversationBufferMemory):
    def __init__(self, k: int = 5, **kwargs: Any):
        super().__init__(**kwargs)
        self.k = k

    def load_memory_variables(self, _: Dict[str, Any]):
        msgs = self.chat_memory.messages[-self.k :]
        if self.return_messages:
            return {self.memory_key: list(msgs)}
        return {self.memory_key: "\n".join(_as_text(m) for m in msgs)}


class ConversationSummaryMemory(ConversationBufferMemory):
    @classmethod
    def from_messages(cls, llm: Any = None, chat_memory: Optional[BaseChatMessageHistory] = None, **kwargs: Any):
        return cls(chat_memory=chat_memory, llm=llm, **kwargs)


class ConversationSummaryBufferMemory(ConversationBufferMemory):
    pass


class ConversationTokenBufferMemory(ConversationBufferMemory):
    pass


class ConversationEntityMemory(ConversationBufferMemory):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.entity_store = type("EntityStore", (), {"store": {}})()

    def load_memory_variables(self, inputs: Dict[str, Any]):
        vars_ = super().load_memory_variables(inputs)
        vars_["entity"] = next(iter(self.entity_store.store.keys()), "")
        return vars_

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        super().save_context(inputs, outputs)
        text = _as_text(inputs)
        for token in ["钢铁侠", "美国队长", "绿巨人", "蜘蛛侠", "纽约", "小明", "小智"]:
            if token in text:
                self.entity_store.store[token] = text


class ConversationKGMemory(ConversationBufferMemory):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.triplets: List[tuple] = []

    def get_knowledge_triplets(self, text: str):
        return self.triplets

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]):
        super().save_context(inputs, outputs)
        text = _as_text(inputs)
        if "是" in text:
            parts = text.split("是", 1)
            if len(parts) == 2:
                self.triplets.append((parts[0].strip(), "是", parts[1].strip()))


class VectorStoreRetrieverMemory(ConversationBufferMemory):
    pass


class BaseRetriever:
    def __init__(self, vectorstore: Any, search_kwargs: Optional[Dict[str, Any]] = None):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {"k": 4}

    def invoke(self, query: str = "", **kwargs: Any):
        if not query:
            query = kwargs.get("input", kwargs.get("query", ""))
        return self.vectorstore.similarity_search(query, k=self.search_kwargs.get("k", 4))


class _SimpleEmbeddings:
    def __init__(self, model: str = "nomic-embed-text", **kwargs: Any):
        self.model = model
        self.kwargs = kwargs

    def _embed(self, text: str) -> List[float]:
        import hashlib

        digest = hashlib.sha256((self.model + "|" + text).encode("utf-8", errors="ignore")).digest()
        if np is None:
            return [byte / 255.0 for byte in digest] * 4
        arr = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        return (arr / 255.0).tolist() * 4

    def embed_query(self, text: str):
        return self._embed(text)

    def embed_documents(self, texts: Iterable[str]):
        return [self._embed(text) for text in texts]


class _VectorStore:
    def __init__(self, docs: List[Document], vectors: List[List[float]], embedding: Any, persist_directory: Optional[str] = None):
        self.docs = docs
        if np is None:
            self.vectors = [list(vector) for vector in vectors] if vectors else []
        else:
            self.vectors = np.array(vectors, dtype=np.float32) if vectors else np.zeros((0, 1), dtype=np.float32)
        self.embedding = embedding
        self.persist_directory = persist_directory

    @classmethod
    def from_documents(cls, documents: List[Document], embedding: Any, persist_directory: Optional[str] = None):
        vectors = embedding.embed_documents([doc.page_content for doc in documents])
        return cls(list(documents), vectors, embedding, persist_directory)

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Any, metadatas: Optional[List[Dict[str, Any]]] = None, persist_directory: Optional[str] = None):
        docs = [Document(text, metadatas[i] if metadatas and i < len(metadatas) else {}) for i, text in enumerate(texts)]
        return cls.from_documents(docs, embedding, persist_directory)

    def similarity_search_by_vector(self, vector: List[float], k: int = 4, filter: Optional[Dict[str, Any]] = None):
        if len(self.docs) == 0:
            return []
        if np is None:
            def _norm(values: List[float]) -> float:
                return sum(value * value for value in values) ** 0.5

            query_norm = _norm(vector) + 1e-8
            scores = []
            for index, stored_vector in enumerate(self.vectors):
                stored_norm = _norm(stored_vector) + 1e-8
                dot = sum(a * b for a, b in zip(stored_vector, vector))
                scores.append((dot / (stored_norm * query_norm), index))
            scores.sort(key=lambda item: item[0], reverse=True)
            idx = [index for _, index in scores[:k]]
        else:
            v = np.array(vector, dtype=np.float32)
            matrix = self.vectors
            denom = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(v) + 1e-8)
            scores = matrix @ v / (denom + 1e-8)
            idx = np.argsort(scores)[::-1][:k]
        docs = [self.docs[i] for i in idx]
        if filter:
            docs = [doc for doc in docs if all(doc.metadata.get(key) == value for key, value in filter.items())]
        return docs

    def similarity_search(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None, **kwargs: Any):
        return self.similarity_search_by_vector(self.embedding.embed_query(query), k=k, filter=filter)

    def similarity_search_with_score(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None):
        docs = self.similarity_search(query, k=k, filter=filter)
        return [(doc, float(i)) for i, doc in enumerate(docs)]

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None, search_type: str = "similarity", **kwargs: Any):
        retriever = BaseRetriever(self, search_kwargs)
        retriever.search_type = search_type
        retriever.kwargs = kwargs
        return retriever

    def persist(self):
        if not self.persist_directory:
            return
        path = Path(self.persist_directory)
        path.mkdir(parents=True, exist_ok=True)
        payload = {"docs": [(d.page_content, d.metadata) for d in self.docs]}
        (path / "store.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


class FAISS(_VectorStore):
    pass


class Chroma(_VectorStore):
    @classmethod
    def from_documents(cls, documents: List[Document], embedding: Any, persist_directory: Optional[str] = None):
        store = super().from_documents(documents, embedding, persist_directory)
        store.persist()
        return store

    def _similarity_search_with_relevance_scores(self, query: str, k: int = 4, filter: Optional[Dict[str, Any]] = None):
        docs = self.similarity_search(query, k=k, filter=filter)
        if not docs:
            return []
        return [(doc, 1.0 / (idx + 1)) for idx, doc in enumerate(docs)]

    def max_marginal_relevance_search(self, query: str, k: int = 4, fetch_k: int = 20, lambda_mult: float = 0.5, filter: Optional[Dict[str, Any]] = None):
        return self.similarity_search(query, k=k, filter=filter)


class SQLDatabase:
    def __init__(self, conn: sqlite3.Connection, dialect: str = "sqlite"):
        self.conn = conn
        self.dialect = dialect

    @classmethod
    def from_uri(cls, uri: str):
        if uri.startswith("sqlite:///"):
            path = uri.replace("sqlite:///", "", 1)
            path_obj = Path(path)
            if path_obj.parent and not path_obj.parent.exists():
                path_obj.parent.mkdir(parents=True, exist_ok=True)
            return cls(sqlite3.connect(path), dialect="sqlite")

        conn = sqlite3.connect(":memory:")
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS employees (id INTEGER PRIMARY KEY, name TEXT, salary INTEGER)")
        cur.executemany(
            "INSERT INTO employees (name, salary) VALUES (?, ?)",
            [("Alice", 12000), ("Bob", 9000), ("Carol", 15000)],
        )
        conn.commit()
        dialect = uri.split(":", 1)[0] if ":" in uri else "sqlite"
        return cls(conn, dialect=dialect)

    def run(self, query: str):
        cur = self.conn.cursor()
        cur.execute(query)
        return cur.fetchall()

    def get_table_info(self):
        cur = self.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return [row[0] for row in cur.fetchall()]

    def get_usable_table_names(self):
        return self.get_table_info()


class TextLoader:
    def __init__(self, file_path: str, encoding: Optional[str] = None):
        self.file_path = file_path
        self.encoding = encoding

    def load(self):
        encodings = [self.encoding] if self.encoding else ["utf-8", "gbk", "utf-8-sig", "latin-1"]
        for enc in encodings:
            if not enc:
                continue
            try:
                text = Path(self.file_path).read_text(encoding=enc)
                return [Document(text, {"source": self.file_path})]
            except Exception:
                continue
        return [Document(Path(self.file_path).read_bytes().decode("utf-8", errors="ignore"), {"source": self.file_path})]


class CSVLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        docs = []
        with open(self.file_path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                docs.append(Document(json.dumps(row, ensure_ascii=False), {"row": i, "source": self.file_path}))
        return docs


class JSONLoader:
    def __init__(self, file_path: str, jq_schema: str = ".", text_content: bool = True):
        self.file_path = file_path
        self.jq_schema = jq_schema
        self.text_content = text_content

    def load(self):
        data = json.loads(Path(self.file_path).read_text(encoding="utf-8", errors="ignore"))
        return [Document(json.dumps(data, ensure_ascii=False), {"source": self.file_path})]


class PyPDFLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self):
        try:
            from pypdf import PdfReader

            reader = PdfReader(self.file_path)
            return [Document(page.extract_text() or "", {"page": i, "source": self.file_path}) for i, page in enumerate(reader.pages)]
        except Exception:
            return [Document(f"PDF loader unavailable for {self.file_path}", {"source": self.file_path})]


class WebBaseLoader:
    def __init__(self, web_path: str, bs_kwargs: Optional[Dict[str, Any]] = None):
        self.web_path = web_path
        self.bs_kwargs = bs_kwargs or {}

    def load(self):
        try:
            with urllib.request.urlopen(self.web_path, timeout=20) as response:
                text = response.read().decode("utf-8", errors="ignore")
            if self.bs_kwargs:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(text, "html.parser")
                parse_only = self.bs_kwargs.get("parse_only")
                if parse_only is not None:
                    soup = BeautifulSoup(text, "html.parser", parse_only=parse_only)
                text = soup.get_text("\n", strip=True)
            return [Document(text, {"source": self.web_path})]
        except Exception:
            return [Document(f"Unable to fetch {self.web_path}", {"source": self.web_path})]


class TextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs: Any):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.kwargs = kwargs

    @classmethod
    def from_tiktoken_encoder(cls, **kwargs: Any):
        return cls(**kwargs)

    def split_text(self, text: str):
        out = []
        start = 0
        while start < len(text):
            out.append(text[start : start + self.chunk_size])
            if start + self.chunk_size >= len(text):
                break
            start += max(1, self.chunk_size - self.chunk_overlap)
        return out

    def split_documents(self, docs: List[Document]):
        out = []
        for doc in docs:
            out.extend(Document(chunk, dict(doc.metadata)) for chunk in self.split_text(doc.page_content))
        return out


class CharacterTextSplitter(TextSplitter):
    pass


class TokenTextSplitter(TextSplitter):
    pass


class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, **kwargs: Any):
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)

    @classmethod
    def from_tiktoken_encoder(cls, **kwargs: Any):
        return cls(**kwargs)

    def split_text(self, text: str):
        return self.splitter.split_text(text)

    def create_documents(self, texts: List[str]):
        return [Document(text, {}) for text in texts]

    def split_documents(self, docs: List[Document]):
        return self.splitter.split_documents(docs)


class SemanticSimilarityExampleSelector:
    def __init__(self, examples: List[Dict[str, Any]], embeddings: Any, vectorstore_cls: Any = None, k: int = 1):
        self.examples = examples
        self.embeddings = embeddings
        self.k = k

    @classmethod
    def from_examples(cls, examples: List[Dict[str, Any]], embeddings: Any, vectorstore_cls: Any = None, k: int = 1, input_keys: Optional[List[str]] = None):
        return cls(examples, embeddings, vectorstore_cls, k)

    def select_examples(self, input_variables: Dict[str, Any]):
        return self.examples[: self.k]


class StructuredTool:
    def __init__(self, name: str, description: str, func: Callable[..., Any], args_schema: Any = None):
        self.name = name
        self.description = description
        self.func = func
        self.args_schema = args_schema
        self.return_direct = False
        self.args = {}
        if args_schema is not None and hasattr(args_schema, "model_fields"):
            self.args = {name: getattr(field, "description", "") for name, field in args_schema.model_fields.items()}
        elif args_schema is not None and hasattr(args_schema, "__fields__"):
            self.args = {name: getattr(field.field_info, "description", "") for name, field in args_schema.__fields__.items()}

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any],
        name: Optional[str] = None,
        description: Optional[str] = None,
        return_direct: bool = False,
        args_schema: Any = None,
    ):
        tool = cls(name or func.__name__, description or (func.__doc__ or ""), func, args_schema=args_schema)
        tool.return_direct = return_direct
        if not tool.args:
            import inspect

            signature = inspect.signature(func)
            tool.args = {
                param_name: getattr(param.annotation, "__name__", str(param.annotation)) if param.annotation is not inspect._empty else "string"
                for param_name, param in signature.parameters.items()
            }
        return tool

    def invoke(self, input: Any):
        if isinstance(input, dict):
            return self.func(**input)
        return self.func(input)

    def run(self, input: Any):
        return self.invoke(input)


def tool(func: Optional[Callable[..., Any]] = None, **kwargs: Any):
    name = kwargs.pop("name", None) or kwargs.pop("name_or_callable", None)
    description = kwargs.pop("description", None)
    return_direct = kwargs.pop("return_direct", False)
    args_schema = kwargs.pop("args_schema", None)

    def wrap(fn):
        wrapped = StructuredTool.from_function(
            func=fn,
            name=name or fn.__name__,
            description=description or (fn.__doc__ or ""),
            return_direct=return_direct,
            args_schema=args_schema,
        )
        return wrapped

    return wrap(func) if callable(func) else wrap


class Tool(StructuredTool):
    pass


def create_retriever_tool(retriever: Any, name: str, description: str):
    def fn(query: str):
        docs = retriever.invoke(query)
        return "\n".join(doc.page_content for doc in docs)

    return Tool(name=name, description=description, func=fn)


def convert_to_openai_function(tool_obj: Any):
    return {"name": getattr(tool_obj, "name", tool_obj.__class__.__name__), "description": getattr(tool_obj, "description", "")}


class RunnableWithMessageHistory(Runnable):
    def __init__(self, runnable: Any, get_session_history: Callable[[str], BaseChatMessageHistory], input_messages_key: str = "input", history_messages_key: str = "history"):
        self.runnable = runnable
        self.get_session_history = get_session_history
        self.input_messages_key = input_messages_key
        self.history_messages_key = history_messages_key

    def invoke(self, input: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        session_id = (config or {}).get("configurable", {}).get("session_id", "default")
        history = self.get_session_history(session_id)
        payload = dict(input)
        payload[self.history_messages_key] = list(history.messages)
        result = self.runnable.invoke(payload) if hasattr(self.runnable, "invoke") else self.runnable(payload)
        history.add_user_message(_as_text(input.get(self.input_messages_key, input)))
        history.add_ai_message(_as_text(result))
        return result


class LLMChain(Runnable):
    def __init__(self, llm: Any, prompt: Any, memory: Optional[Any] = None, verbose: bool = False, **kwargs: Any):
        self.llm = llm
        self.prompt = prompt
        self.memory = memory
        self.verbose = verbose
        self.kwargs = kwargs
        self.output_key = kwargs.get("output_key", "text")

    def invoke(self, input: Dict[str, Any]):
        payload = dict(input) if isinstance(input, dict) else {"input": input}
        if self.memory is not None and hasattr(self.memory, "load_memory_variables"):
            payload = {**self.memory.load_memory_variables({}), **payload}
        formatted = self.prompt.invoke(payload) if hasattr(self.prompt, "invoke") else self.prompt.format(**payload)
        result = self.llm.invoke(formatted)
        text = _as_text(result)
        if self.memory is not None and hasattr(self.memory, "save_context"):
            self.memory.save_context(input, {"output": text})
        return {self.output_key: text, "text": text, "output": text}


class SimpleSequentialChain(Runnable):
    def __init__(self, chains: List[Any], verbose: bool = False, **kwargs: Any):
        self.chains = chains
        self.verbose = verbose
        self.kwargs = kwargs

    def _extract_output(self, value: Any):
        if isinstance(value, dict):
            if "text" in value:
                return value["text"]
            if "output" in value:
                return value["output"]
            if "input" in value and len(value) == 1:
                return value["input"]
        return value

    def _infer_input_key(self, chain: Any) -> str:
        prompt = getattr(chain, "prompt", None)
        if prompt is None:
            return "input"
        input_variables = getattr(prompt, "input_variables", None) or []
        if input_variables:
            return input_variables[0]
        messages = getattr(prompt, "messages", None) or []
        for item in messages:
            if isinstance(item, tuple) and len(item) == 2:
                fields = re.findall(r"{([^{}]+)}", item[1])
                if fields:
                    return fields[0]
        return "input"

    def invoke(self, input: Any):
        value = self._extract_output(input)
        for chain in self.chains:
            chain_input = {self._infer_input_key(chain): self._extract_output(value)}
            value = chain.invoke(chain_input) if hasattr(chain, "invoke") else chain(self._extract_output(value))
        return value


class SequentialChain(SimpleSequentialChain):
    def __init__(self, chains: List[Any], input_variables: Optional[List[str]] = None, output_variables: Optional[List[str]] = None, verbose: bool = False, **kwargs: Any):
        super().__init__(chains=chains, verbose=verbose, **kwargs)
        self.input_variables = input_variables or []
        self.output_variables = output_variables or []

    def invoke(self, input: Any):
        if isinstance(input, dict):
            payload = dict(input)
        else:
            key = self.input_variables[0] if self.input_variables else "input"
            payload = {key: input}
        for chain in self.chains:
            result = chain.invoke(payload) if hasattr(chain, "invoke") else chain(payload)
            if isinstance(result, dict):
                payload.update(result)
            else:
                payload[getattr(chain, "output_key", "output")] = result
        if self.output_variables:
            return {key: payload.get(key) for key in self.output_variables}
        return payload


class ConversationChain(Runnable):
    def __init__(self, llm: Any, memory: Optional[Any] = None, prompt: Optional[Any] = None):
        self.llm = llm
        self.memory = memory or ConversationBufferMemory()
        self.prompt = prompt or PromptTemplate.from_template("{input}")

    def invoke(self, input: Dict[str, Any]):
        vars_ = self.memory.load_memory_variables({})
        payload = dict(vars_)
        payload.update(input)
        text = self.prompt.invoke(payload) if hasattr(self.prompt, "invoke") else self.prompt.format(**payload)
        result = self.llm.invoke(text)
        self.memory.save_context(input, {"output": _as_text(result)})
        return result


class AgentType:
    ZERO_SHOT_REACT_DESCRIPTION = "zero_shot_react"
    OPENAI_FUNCTIONS = "openai_functions"
    CONVERSATIONAL_REACT_DESCRIPTION = "conversational_react"


class _SimpleAgent:
    def __init__(self, llm: Any, tools: List[Any], prompt: Any = None):
        self.llm = llm
        self.tools = tools
        self.prompt = prompt

    def invoke(self, input: Dict[str, Any]):
        query = input.get("input", input if isinstance(input, str) else str(input))
        tool_text = ""
        for tool_obj in self.tools:
            try:
                tool_text += f"\n[{getattr(tool_obj, 'name', 'tool')}]\n{_as_text(tool_obj.invoke(query))}"
            except Exception as exc:
                tool_text += f"\n[{getattr(tool_obj, 'name', 'tool')}] error: {exc}"
        prompt_input = query if not tool_text else f"{query}\n\n{tool_text}"
        return self.llm.invoke(prompt_input)


class AgentExecutor(Runnable):
    def __init__(self, agent: Any, tools: List[Any], verbose: bool = False, **kwargs: Any):
        self.agent = agent
        self.tools = tools
        self.verbose = verbose
        self.kwargs = kwargs

    def invoke(self, input: Any):
        payload = input if isinstance(input, dict) else {"input": input}
        if hasattr(self.agent, "invoke"):
            return self.agent.invoke(payload)
        return _SimpleAgent(self.kwargs.get("llm"), self.tools).invoke(payload)


def initialize_agent(*, tools: List[Any], llm: Any, agent: Any, verbose: bool = False, **kwargs: Any):
    return AgentExecutor(agent=_SimpleAgent(llm, tools), tools=tools, verbose=verbose, llm=llm, **kwargs)


def create_openai_functions_agent(llm: Any, tools: List[Any], prompt: Any):
    return _SimpleAgent(llm, tools, prompt)


def create_react_agent(llm: Any, tools: List[Any], prompt: Any):
    return _SimpleAgent(llm, tools, prompt)


def create_tool_calling_agent(llm: Any, tools: List[Any], prompt: Any):
    return _SimpleAgent(llm, tools, prompt)


def create_stuff_documents_chain(llm: Any, prompt: Any, document_variable_name: str = "context", **kwargs: Any):
    class _Chain(Runnable):
        def invoke(self, input: Dict[str, Any]):
            payload = dict(input)
            docs = payload.get(document_variable_name, [])
            payload[document_variable_name] = "\n".join(getattr(doc, "page_content", str(doc)) for doc in docs)
            return llm.invoke(prompt.invoke(payload) if hasattr(prompt, "invoke") else prompt.format(**payload))

    return _Chain()


def create_sql_query_chain(llm: Any, db: Any, prompt: Any = None):
    class _Chain(Runnable):
        def invoke(self, input: Dict[str, Any]):
            question = input.get("question") or input.get("input") or str(input)
            tables = db.get_table_info() if hasattr(db, "get_table_info") else []
            return llm.invoke(f"Question: {question}\nTables: {tables}")

    return _Chain()


class PythonREPL:
    def __init__(self):
        self.locals: Dict[str, Any] = {}

    def run(self, command: str):
        try:
            buf = []
            exec(command, {}, self.locals)
            return ""
        except Exception as exc:
            return str(exc)


class _Hub:
    def pull(self, name: str):
        return ChatPromptTemplate.from_messages([("system", f"hub:{name}"), ("user", "{input}")])


hub = _Hub()


def _register_core_submodules():
    _register(__name__ + ".messages", {"BaseMessage": BaseMessage, "ChatMessage": ChatMessage, "HumanMessage": HumanMessage, "SystemMessage": SystemMessage, "AIMessage": AIMessage}, is_package=True)
    _register(__name__ + ".messages.ai", {"AIMessage": AIMessage})
    _register(__name__ + ".prompts", {"PromptTemplate": PromptTemplate, "ChatPromptTemplate": ChatPromptTemplate, "MessagesPlaceholder": MessagesPlaceholder, "FewShotPromptTemplate": FewShotPromptTemplate, "FewShotChatMessagePromptTemplate": FewShotChatMessagePromptTemplate, "HumanMessagePromptTemplate": HumanMessagePromptTemplate, "SystemMessagePromptTemplate": SystemMessagePromptTemplate, "load_prompt": load_prompt}, is_package=True)
    _register(__name__ + ".prompts.prompt", {"PromptTemplate": PromptTemplate})
    _register(__name__ + ".prompts.chat", {"ChatPromptTemplate": ChatPromptTemplate, "MessagesPlaceholder": MessagesPlaceholder, "HumanMessagePromptTemplate": HumanMessagePromptTemplate, "SystemMessagePromptTemplate": SystemMessagePromptTemplate})
    _register(__name__ + ".output_parsers", {"StrOutputParser": StrOutputParser, "JsonOutputParser": JsonOutputParser, "XMLOutputParser": XMLOutputParser, "CommaSeparatedListOutputParser": CommaSeparatedListOutputParser, "DatetimeOutputParser": DatetimeOutputParser}, is_package=True)
    _register(__name__ + ".output_parsers.xml", {"XMLOutputParser": XMLOutputParser})
    _register(__name__ + ".documents", {"Document": Document})
    _register(__name__ + ".chat_history", {"BaseChatMessageHistory": BaseChatMessageHistory})
    _register(__name__ + ".example_selectors", {"SemanticSimilarityExampleSelector": SemanticSimilarityExampleSelector})
    _register(__name__ + ".tools", {"StructuredTool": StructuredTool, "tool": tool, "Tool": Tool, "create_retriever_tool": create_retriever_tool}, is_package=True)
    _register(__name__ + ".tools.structured", {"StructuredTool": StructuredTool})
    _register(__name__ + ".utils", {"pre_init": lambda fn: fn, "convert_to_openai_function": convert_to_openai_function}, is_package=True)
    _register(__name__ + ".utils.function_calling", {"convert_to_openai_function": convert_to_openai_function})
    _register(__name__ + ".runnables", {}, is_package=True)
    _register(__name__ + ".runnables.history", {"RunnableWithMessageHistory": RunnableWithMessageHistory})


_register_core_submodules()
