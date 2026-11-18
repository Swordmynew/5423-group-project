"""The four tools exposed to the LangGraph agent."""

from tools.document_reader import document_reader
from tools.metadata_filter import metadata_filter
from tools.retrieval_tool import unified_retrieval
from tools.text_searcher import text_searcher

# Exactly one retrieval tool + three supporting tools.
TOOLS = [unified_retrieval, document_reader, text_searcher, metadata_filter]
TOOL_MAP = {tool.name: tool for tool in TOOLS}

__all__ = [
    "TOOLS",
    "TOOL_MAP",
    "unified_retrieval",
    "document_reader",
    "text_searcher",
    "metadata_filter",
]
