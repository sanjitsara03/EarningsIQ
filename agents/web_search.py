# The web search agent runs when a question needs current information that isn't in SEC filings
# things like recent news, analyst sentiment, or current stock price.
# It uses Tavily as its only tool and returns a plain text summary.

import logging
import os

from dotenv import load_dotenv
from tavily import TavilyClient

from llm import (
    LoopStallError,
    assistant_msg_from_response,
    chat,
    to_openai_tools,
    tool_result_msgs,
    traced,
    user_msg,
)

load_dotenv()

logger = logging.getLogger(__name__)

MAX_TURNS = 5

SYSTEM_PROMPT = """You are a financial research assistant with access to a web search tool.

Use the search tool to find relevant, current information to answer the user's question.
Synthesize the results into a concise, factual summary. Cite sources where relevant.
Focus on information that would be useful for investment research."""

# Tavily tool schema (provider-neutral flat shape; converted to OpenAI format below).
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


OPENAI_TOOLS = to_openai_tools([TAVILY_TOOL])


# Standard tool loop that runs until the LLM stops calling tools and returns a text summary.
# At MAX_TURNS, one final call with tools disabled forces a text answer.
@traced("web_search")
def run_web_search(query: str) -> str:
    messages = [user_msg(query)]

    for _ in range(MAX_TURNS):
        resp = chat("web_search", messages, system=SYSTEM_PROMPT, tools=OPENAI_TOOLS)

        # No tool calls means the LLM is done — return the text summary.
        if not resp.tool_calls:
            return resp.text or ""

        pairs = []
        for tc in resp.tool_calls:
            logger.info(f"Web search query: {tc.args['query']!r}")
            pairs.append((tc.id, _run_tavily_search(tc.args["query"])))

        messages.append(assistant_msg_from_response(resp))
        messages.extend(tool_result_msgs(pairs))

    # Turn cap reached — force a final text summary from the results gathered so far.
    resp = chat("web_search", messages, system=SYSTEM_PROMPT, tools=OPENAI_TOOLS, tool_choice="none")
    if resp.text and resp.text.strip():
        return resp.text
    raise LoopStallError("web_search", f"no text answer after {MAX_TURNS} turns")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    query = " ".join(sys.argv[1:]) or "What is the latest news on NVIDIA?"
    print(run_web_search(query))
