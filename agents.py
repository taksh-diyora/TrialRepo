import os
import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
)
from langchain_cohere import ChatCohere
from langgraph.graph import StateGraph, END

from tools import tavily_tool, rag_tool


# ============ LLM ============
llm = ChatCohere(
    model="c4ai-aya-23-8b",          # Stable, free Cohere model
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    temperature=0,
)


# ============ STATE ============
class AgentState(TypedDict):
    # LangGraph will keep appending messages from each node
    messages: Annotated[Sequence[BaseMessage], operator.add]
    # Used only by Supervisor to decide routing
    next: str


# ============ SUPERVISOR ============
def supervisor_node(state: AgentState):
    """Route based ONLY on the latest HUMAN message."""

    # Filter only HumanMessage objects
    human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    if not human_messages:
        return {"next": "FINISH"}

    query = human_messages[-1].content.lower().strip()

    # Stop conditions
    if query in ["quit", "exit", "bye"]:
        return {"next": "FINISH"}

    # PDF-related routing
    pdf_keywords = ["pdf", "document", "file", "page", "pages"]
    if any(word in query for word in pdf_keywords):
        return {"next": "PDF_Analyst"}

    # Default: web search
    return {"next": "Web_Searcher"}


# ============ WEB SEARCHER ============
def web_search_node(state: AgentState):
    """Uses Tavily + LLM to answer general web questions with sources."""

    query = state["messages"][-1].content

    # Call Tavily search (langchain_tavily.TavilySearch)
    search_results = tavily_tool.invoke({"query": query})

    prompt = f"""
You are a helpful assistant. Use ONLY the following web search results to answer.

QUESTION:
{query}

SEARCH RESULTS (JSON-like):
{search_results}

Respond in this format:

- **Answer:** <short, accurate answer in 2–4 sentences>
- **Sources:**
  - <source 1 name or domain + URL if available>
  - <source 2 name or domain + URL if available>
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "messages": [
            AIMessage(content=response.content, name="Web_Searcher")
        ]
    }


# ============ PDF ANALYST ============
def pdf_analyst_node(state: AgentState):
    """Stub/real PDF agent using rag_tool. Currently just echoes a stub message."""

    query = state["messages"][-1].content

    # We design rag_tool to accept a simple string query
    result = rag_tool.invoke(query)

    return {
        "messages": [
            AIMessage(content=str(result), name="PDF_Analyst")
        ]
    }


# ============ GRAPH BUILDING ============
workflow = StateGraph(AgentState)

# Nodes
workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Web_Searcher", web_search_node)
workflow.add_node("PDF_Analyst", pdf_analyst_node)

# Entry point
workflow.set_entry_point("Supervisor")

# Routing from Supervisor based on `next`
workflow.add_conditional_edges(
    "Supervisor",
    lambda state: state["next"],
    {
        "Web_Searcher": "Web_Searcher",
        "PDF_Analyst": "PDF_Analyst",
        "FINISH": END,
    },
)

# After either worker, we end the run (one answer per user question)
workflow.add_edge("Web_Searcher", END)
workflow.add_edge("PDF_Analyst", END)

graph = workflow.compile()
