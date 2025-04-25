import os
from tavily import TavilyClient
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Initialize Tavily client
client = TavilyClient(api_key=TAVILY_API_KEY)

# Research function
def research_topic(query: str, num_results: int = 5) -> str:
    print(f"🔍 Researching: {query}")
    
    results = client.search(query=query, search_depth="advanced", max_results=num_results)

    # Extract and format results
    structured_data = ""
    for i, result in enumerate(results.get("results", []), 1):
        title = result.get("title", "No Title")
        content = result.get("content", "No Content")
        url = result.get("url", "")
        
        structured_data += f"{i}. {title}\n{content}\n(Source: {url})\n\n"

    return structured_data.strip()
