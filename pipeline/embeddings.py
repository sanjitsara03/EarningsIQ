# Shared embedding model for the whole app. Ingestion (documents) and the Comparison Agent
# (queries) MUST use the same model — vectors from different models live in different spaces,
# and mixing them makes cosine similarity silently meaningless rather than raising an error.
import os

from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings

load_dotenv()

# voyage-finance-2: finance-domain embeddings (SEC filings, earnings), 1024 dims.
EMBED_MODEL = "voyage-finance-2"
EMBED_DIM = 1024

# Lazily constructed so importing this module never requires VOYAGE_API_KEY
# (mirrors the lazy DB pool — RQ workers import before env is guaranteed).
_embed_model: VoyageAIEmbeddings | None = None


def _get_model() -> VoyageAIEmbeddings:
    global _embed_model
    if _embed_model is None:
        if not os.getenv("VOYAGE_API_KEY"):
            raise RuntimeError("VOYAGE_API_KEY is not set — required for embeddings (voyage-finance-2).")
        _embed_model = VoyageAIEmbeddings(model=EMBED_MODEL, batch_size=32)
    return _embed_model


# Embeds document chunks for storage in pgvector.
def embed_documents(texts: list[str]) -> list[list[float]]:
    return _get_model().embed_documents(texts)


# Embeds a search query. Voyage uses a distinct query input type under the hood for better retrieval.
def embed_query(text: str) -> list[float]:
    return _get_model().embed_query(text)
