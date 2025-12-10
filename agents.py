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


# ===================== LLM =====================
llm = ChatCohere(
    model="c4ai-aya-23-8b",
    cohere_api_key=os.getenv("COHERE_API_KEY"),
    temperature=0
)


# ===================== MEMORY =====================
MEMORY = []           # stores entire conversation
MEMORY_LIMIT = 10     # keep last N pairs


def add_to_memory(message: BaseMessage):
    """Store user + AI messages, but limit size."""
    MEMORY.append(message)
    if len(MEMORY) > MEMORY_LIMIT:
        MEMORY.pop(0)


def get_memory():
    """Return memory messages for workers."""
    return MEMORY[-MEMORY_LIMIT:]


# ===================== STATE =====================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


# ===================== SUPERVISOR =====================
def supervisor_node(state: AgentState):
    """
    IMPORTANT:
    Supervisor should NOT see memory.
    It should ONLY react to the latest USER message.
    """
    last_msg = state["messages"][-1]

    if isinstance(last_msg, AIMessage):
        # AI messages are NOT routing triggers → go straight to FINISH
        return {"next": "FINISH"}

    query = last_msg.content.lower()

    if any(w in query for w in ["pdf", "document", "file", "page"]):
        return {"next": "PDF_Analyst"}

    if query in ["quit", "exit", "bye"]:
        return {"next": "FINISH"}

    return {"next": "Web_Searcher"}


# ===================== WEB SEARCH WORKER =====================
def web_search_node(state: AgentState):
    query = state["messages"][-1].content

    # Worker gets memory
    memory_msgs = get_memory()

    results = tavily_tool.invoke({"query": query})

    prompt = f"""
    You are a factual answering agent.

    Using ONLY the verified search results below + conversation memory,
    produce a clear and well-structured answer.

    === MEMORY ===
    {memory_msgs}

    === QUESTION ===
    {query}

    === SEARCH RESULTS ===
    {results}

    FORMAT YOUR ANSWER EXACTLY LIKE THIS:

    **Answer:**  
    <2–5 sentence factual answer>

    **Sources:**  
    - <source 1 name or domain>  
    - <source 2 name or domain>  
    - <source 3 name or domain>  

    Make sure:
    - Answer is short but accurate  
    - Sources list ONLY real domains found in the search results  
    - No hallucinated sources  
    """


    response = llm.invoke([HumanMessage(content=prompt)])
    ai_msg = AIMessage(content=response.content, name="Web_Searcher")

    add_to_memory(HumanMessage(content=query))
    add_to_memory(ai_msg)

    return {
        "messages": [ai_msg],
        "next": "Supervisor"
    }


# ===================== PDF ANALYST WORKER =====================
def pdf_analyst_node(state: AgentState):
    query = state["messages"][-1].content

    memory_msgs = get_memory()

    results = rag_tool.invoke({"query": query})
    final_answer = f"""
    Using conversation memory + PDF retrieval:
    MEMORY: {memory_msgs}
    RESULT: {results}
    """

    ai_msg = AIMessage(content=final_answer, name="PDF_Analyst")

    add_to_memory(HumanMessage(content=query))
    add_to_memory(ai_msg)

    return {
        "messages": [ai_msg],
        "next": "Supervisor"
    }


# ===================== GRAPH =====================
workflow = StateGraph(AgentState)

workflow.add_node("Supervisor", supervisor_node)
workflow.add_node("Web_Searcher", web_search_node)
workflow.add_node("PDF_Analyst", pdf_analyst_node)

workflow.set_entry_point("Supervisor")

workflow.add_conditional_edges(
    "Supervisor",
    lambda x: x["next"],
    {
        "Web_Searcher": "Web_Searcher",
        "PDF_Analyst": "PDF_Analyst",
        "FINISH": END,
    }
)

workflow.add_edge("Web_Searcher", "Supervisor")
workflow.add_edge("PDF_Analyst", "Supervisor")

graph = workflow.compile()
