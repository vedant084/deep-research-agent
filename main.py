from workflows.langgraph_flow import build_graph

# Build the LangGraph
graph = build_graph()

def main():
    print("Deep Research Agent | Powered by LangGraph + Tavily + HuggingFace")
    research_question = input("Enter your research query: ")

    # Initial state for LangGraph
    state = {
        "research_question": research_question,
        "search_results": None,
        "draft": None,
    }

    # Execute the graph
    final_state = graph.invoke(state)

    # Output the final drafted answer
    print("\nFinal Drafted Answer:\n")
    print(final_state["drafted_answer"])

if __name__ == "__main__":
    main()
