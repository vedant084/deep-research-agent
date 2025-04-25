from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

# Import our agents
from agents.researcher_agent import research_topic
from agents.drafter_agent import generate_draft

from typing import TypedDict

# Define graph state
class GraphState(TypedDict):
    research_question: str
    search_results: list[str]
    structured_summary: str
    drafted_answer: str

# Node 1: Researcher Agent
def research_node(state):
    query = state["research_question"]
    research_summary = research_topic(query)
    return {
        "structured_summary": research_summary
    }


# Node 2: Drafter Agent
def draft_node(state):
    research_summary = state["structured_summary"]
    draft = generate_draft(research_summary)
    return {
        "drafted_answer": draft
    }


# Define the graph using LangGraph
def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("research", research_node)
    builder.add_node("draft", draft_node)

    builder.set_entry_point("research")
    builder.add_edge("research", "draft")
    builder.add_edge("draft", END)

    return builder.compile()


