import os
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool

load_dotenv()

# ============ WEB SEARCH (TAVILY) ============
# Uses TAVILY_API_KEY from .env automatically
tavily_tool = TavilySearch(max_results=3)


# ============ PDF / RAG STUB TOOL ============
def retrieve_stub(query: str) -> str:
    """
    Temporary PDF search stub.
    Later you can replace this with a real vector DB lookup.
    """
    return f"📄 PDF search is not configured yet. (Received query: '{query}')"


rag_tool = Tool(
    name="pdf_search_tool",
    func=retrieve_stub,   # accepts a plain string
    description="Search within uploaded PDF documents (currently returns a stub response).",
)
