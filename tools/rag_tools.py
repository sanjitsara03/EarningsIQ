# Tool schemas, handlers, and vector search utilities for the Comparison Agent.
# The four tools are: resolve_filing_ids, vector_search_chunks, fetch_structured_signals, cite_and_answer.
# cite_and_answer is the terminal tool — the agent loop exits when it is called.


from psycopg2.extensions import connection

from db.queries import get_filing_ids_for_ticker, get_signals_for_filings
from pipeline.embeddings import embed_query as _embed_query

TOOLS = [
    {
        "name": "resolve_filing_ids",
        "description": "Look up internal filing IDs for a ticker and filing type. Call this first before searching chunks or fetching signals.",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker, e.g. 'MSFT'."},
                "filing_type": {"type": "string", "enum": ["10-Q", "10-K"]},
                "limit": {"type": "integer", "description": "Number of most recent filings to retrieve. Default 4.", "default": 4},
            },
            "required": ["ticker", "filing_type"],
        },
    },
    {
        "name": "vector_search_chunks",
        "description": "Semantic search over filing chunks using the query text. Returns the most relevant passages for the question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to embed and match against chunks."},
                "filing_ids": {"type": "array", "items": {"type": "integer"}, "description": "Filing IDs to search within."},
                "top_k": {"type": "integer", "description": "Number of chunks to return. Default 8.", "default": 8},
            },
            "required": ["query", "filing_ids"],
        },
    },
    {
        "name": "fetch_structured_signals",
        "description": "Fetch numeric signals (revenue, margins, guidance, segments) across multiple filing periods for comparison.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filing_ids": {"type": "array", "items": {"type": "integer"}, "description": "Filing IDs to fetch signals for."},
            },
            "required": ["filing_ids"],
        },
    },
    {
        "name": "cite_and_answer",
        "description": "Return the final answer with citations. Call this when you have enough evidence to answer the question. This ends the analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string", "description": "The full answer to the user's question."},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "period": {"type": "string"},
                            "section": {"type": "string"},
                            "quote": {"type": "string"},
                        },
                        "required": ["period", "quote"],
                    },
                    "description": "Supporting evidence from the filings.",
                },
            },
            "required": ["answer", "citations"],
        },
    },
]


# Embeds a query string using the shared model (voyage-finance-2). Returns a list of 1024 floats.
def embed_query(text: str) -> list[float]:
    return _embed_query(text)


# Runs pgvector cosine similarity search over chunks for the given filing IDs. Returns top-k results.
def vector_search(
    conn: connection,
    query_embedding: list[float],
    filing_ids: list[int],
    top_k: int = 8,
) -> list[dict]:
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT filing_id, section, content
            FROM chunks
            WHERE filing_id = ANY(%s)
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (filing_ids, embedding_str, top_k),
        )
        return [
            {"filing_id": row[0], "section": row[1], "content": row[2]}
            for row in cur.fetchall()
        ]


# Tool handlers — each receives the LLM's input dict and a db connection, returns data for the LLM.

def handle_resolve_filing_ids(input: dict, conn: connection) -> list[int]:
    return get_filing_ids_for_ticker(conn, input["ticker"], input["filing_type"], input.get("limit", 4))


def handle_vector_search_chunks(input: dict, conn: connection) -> list[dict]:
    embedding = embed_query(input["query"])
    return vector_search(conn, embedding, input["filing_ids"], input.get("top_k", 8))


def handle_fetch_structured_signals(input: dict, conn: connection) -> list[dict]:
    return get_signals_for_filings(conn, input["filing_ids"])


def handle_cite_and_answer(input: dict) -> dict:
    return input
