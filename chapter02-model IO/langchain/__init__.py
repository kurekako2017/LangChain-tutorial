import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core import (
    AgentExecutor,
    AgentType,
    BaseChatMessageHistory,
    ChatMessageHistory,
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationChain,
    ConversationEntityMemory,
    ConversationKGMemory,
    ConversationSummaryBufferMemory,
    ConversationSummaryMemory,
    ConversationTokenBufferMemory,
    Document,
    FAISS,
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    LLMChain,
    PromptTemplate,
    RecursiveCharacterTextSplitter,
    RunnableWithMessageHistory,
    SQLDatabase,
    SequentialChain,
    SimpleSequentialChain,
    StrOutputParser,
    XMLOutputParser,
    StructuredTool,
    Tool,
    Chroma,
    ChatPromptTemplate,
    CharacterTextSplitter,
    FewShotChatMessagePromptTemplate,
    FewShotPromptTemplate,
    HumanMessage,
    AIMessage,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    DatetimeOutputParser,
    RecursiveCharacterTextSplitter,
    SystemMessagePromptTemplate,
    SystemMessage,
    TokenTextSplitter,
    load_prompt,
    create_openai_functions_agent,
    create_react_agent,
    create_sql_query_chain,
    create_stuff_documents_chain,
    create_tool_calling_agent,
    create_retriever_tool,
    hub,
    initialize_agent,
    tool,
)
from langchain_community.document_loaders import (
    CSVLoader,
    DirectoryLoader,
    JSONLoader,
    PyPDFLoader,
    PythonLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    WebBaseLoader,
)
from langchain_openai import ChatOpenAI

__version__ = "0.3.25-local"


def _register(name: str, attrs: Dict[str, Any], is_package: bool = False):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    if is_package:
        module.__path__ = []
    sys.modules[name] = module
    return module


schema_mod = _register(__name__ + ".schema", {"AIMessage": AIMessage, "HumanMessage": HumanMessage, "SystemMessage": SystemMessage})
tools_mod = _register(__name__ + ".tools", {"Tool": Tool})
prompts_mod = _register(
    __name__ + ".prompts",
    {
        "PromptTemplate": PromptTemplate,
        "ChatPromptTemplate": ChatPromptTemplate,
        "MessagesPlaceholder": MessagesPlaceholder,
        "FewShotPromptTemplate": FewShotPromptTemplate,
        "FewShotChatMessagePromptTemplate": FewShotChatMessagePromptTemplate,
        "HumanMessagePromptTemplate": HumanMessagePromptTemplate,
        "SystemMessagePromptTemplate": SystemMessagePromptTemplate,
        "load_prompt": load_prompt,
    },
    is_package=True,
)
_register(__name__ + ".prompts.prompt", {"PromptTemplate": PromptTemplate})
_register(__name__ + ".prompts.chat", {"ChatPromptTemplate": ChatPromptTemplate, "MessagesPlaceholder": MessagesPlaceholder, "HumanMessagePromptTemplate": HumanMessagePromptTemplate, "SystemMessagePromptTemplate": SystemMessagePromptTemplate})
_register(__name__ + ".prompts.few_shot", {"FewShotPromptTemplate": FewShotPromptTemplate, "FewShotChatMessagePromptTemplate": FewShotChatMessagePromptTemplate})
_register(
    __name__ + ".output_parsers",
    {
        "StrOutputParser": StrOutputParser,
        "JsonOutputParser": JsonOutputParser,
        "XMLOutputParser": XMLOutputParser,
        "CommaSeparatedListOutputParser": CommaSeparatedListOutputParser,
        "DatetimeOutputParser": DatetimeOutputParser,
    },
    is_package=True,
)
_register(__name__ + ".output_parsers.xml", {"XMLOutputParser": XMLOutputParser})
_register(__name__ + ".text_splitter", {"CharacterTextSplitter": CharacterTextSplitter, "TokenTextSplitter": TokenTextSplitter, "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter})
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
)
memory_mod = _register(
    __name__ + ".memory",
    {
        "ChatMessageHistory": ChatMessageHistory,
        "ConversationBufferMemory": ConversationBufferMemory,
        "ConversationBufferWindowMemory": ConversationBufferWindowMemory,
        "ConversationSummaryMemory": ConversationSummaryMemory,
        "ConversationSummaryBufferMemory": ConversationSummaryBufferMemory,
        "ConversationTokenBufferMemory": ConversationTokenBufferMemory,
        "ConversationEntityMemory": ConversationEntityMemory,
        "ConversationKGMemory": ConversationKGMemory,
        "VectorStoreRetrieverMemory": ConversationBufferMemory,
    },
)
_register(
    __name__ + ".memory.prompt",
    {
        "ENTITY_MEMORY_CONVERSATION_TEMPLATE": "Current conversation:\n{history}\n\nCurrent entity: {entity}\n\nRespond to the user.",
    },
    is_package=False,
)
chains_mod = _register(
    __name__ + ".chains",
    {
        "LLMChain": LLMChain,
        "SequentialChain": SequentialChain,
        "SimpleSequentialChain": SimpleSequentialChain,
        "ConversationChain": ConversationChain,
        "create_sql_query_chain": create_sql_query_chain,
        "create_stuff_documents_chain": create_stuff_documents_chain,
    },
    is_package=True,
)
_register(__name__ + ".chains.llm", {"LLMChain": LLMChain})
_register(__name__ + ".chains.sequential", {"SimpleSequentialChain": SimpleSequentialChain})
_register(__name__ + ".chains.conversation.base", {"ConversationChain": ConversationChain, "LLMChain": LLMChain})
_register(__name__ + ".chains.combine_documents", {"create_stuff_documents_chain": create_stuff_documents_chain})
agents_mod = _register(
    __name__ + ".agents",
    {
        "AgentExecutor": AgentExecutor,
        "AgentType": AgentType,
        "StructuredTool": StructuredTool,
        "Tool": Tool,
        "create_openai_functions_agent": create_openai_functions_agent,
        "create_react_agent": create_react_agent,
        "create_tool_calling_agent": create_tool_calling_agent,
        "initialize_agent": initialize_agent,
    },
    is_package=True,
)
_register(__name__ + ".tools.retriever", {"create_retriever_tool": create_retriever_tool})
_register(__name__ + ".chat_models", {"ChatOpenAI": ChatOpenAI})
_register(__name__ + ".hub", {"pull": hub.pull})

__all__ = [
    "AgentExecutor",
    "AgentType",
    "ChatMessageHistory",
    "ConversationBufferMemory",
    "ConversationBufferWindowMemory",
    "ConversationChain",
    "ConversationEntityMemory",
    "ConversationKGMemory",
    "ConversationSummaryBufferMemory",
    "ConversationSummaryMemory",
    "ConversationTokenBufferMemory",
    "Document",
    "FAISS",
    "JsonOutputParser",
    "LLMChain",
    "PromptTemplate",
    "FewShotPromptTemplate",
    "FewShotChatMessagePromptTemplate",
    "MessagesPlaceholder",
    "HumanMessagePromptTemplate",
    "SystemMessagePromptTemplate",
    "CharacterTextSplitter",
    "TokenTextSplitter",
    "RecursiveCharacterTextSplitter",
    "RunnableWithMessageHistory",
    "SQLDatabase",
    "SequentialChain",
    "SimpleSequentialChain",
    "StrOutputParser",
    "StructuredTool",
    "Tool",
    "Chroma",
    "ChatPromptTemplate",
    "HumanMessage",
    "AIMessage",
    "SystemMessage",
    "hub",
    "initialize_agent",
    "tool",
]
