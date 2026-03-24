# The web search agent runs when a question needs current information that isn't in SEC filings
# things like recent news, analyst sentiment, or current stock price.
# It uses Tavily as its only tool and returns a plain text summary.

import logging
import os

import anthropic
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"

SYSTEM_PROMPT = """You are a financial research assistant with access to a web search tool.

Use the search tool to find relevant, current information to answer the user's question.
Synthesize the results into a concise, factual summary. Cite sources where relevant.
Focus on information that would be useful for investment research."""

# Tavily tool schema for the Anthropic SDK.
TAVILY_TOOL = {
    "name": "tavily_search",
    "description": "Search the web for current financial news, analyst sentiment, stock data, or recent events.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to run.",
            }
        },
        "required": ["query"],
    },
}


# Executes a Tavily search and formats results as a readable string for the LLM.
def _run_tavily_search(query: str) -> str:
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = client.search(query=query, max_results=5)
    results = response.get("results", [])
    if not results:
        return "No results found."
    lines = []
    for r in results:
        lines.append(f"[{r['title']}]({r['url']})\n{r['content']}\n")
    return "\n".join(lines)


# Standard tool loop that runs until the LLM stops calling tools and returns a text summary.
def run_web_search(query: str) -> str:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [{"role": "user", "content": query}]

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[TAVILY_TOOL],
            messages=messages,
        )

        # If no tool calls, the LLM is done — extract and return the text summary.
        if response.stop_reason == "end_turn":
            return next(b.text for b in response.content if hasattr(b, "text"))

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        tool_results = []
        for tool_use in tool_uses:
            logger.info(f"Web search query: {tool_use.input['query']!r}")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": _run_tavily_search(tool_use.input["query"]),
            })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    query = " ".join(sys.argv[1:]) or "What is the latest news on NVIDIA?"
    print(run_web_search(query))
