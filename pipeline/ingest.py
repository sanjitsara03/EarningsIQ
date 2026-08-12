import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from db.connection import get_connection
from db.queries import filing_exists, insert_filing, update_filing_status
from pipeline.embeddings import embed_documents
from scraper.edgar_client import fetch_filing, get_10k_filings, get_10q_filings, get_cik
from scraper.parser import parse_sections

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)


# Insert a batch of chunks (with embeddings) into the chunks table.
# Embedding is stringified to the format pgvector expects.
def _insert_chunks(conn, filing_id: int, chunks: list[dict]) -> None:
    with conn.cursor() as cur:
        for chunk in chunks:
            embedding_str = "[" + ",".join(str(x) for x in chunk["embedding"]) + "]"
            cur.execute(
                """
                INSERT INTO chunks (filing_id, section, chunk_index, content, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                """,
                (filing_id, chunk["section"], chunk["chunk_index"], chunk["content"], embedding_str),
            )


# Core pipeline for a single filing: chunk → embed → insert → update status.
# Returns the new filing_id, or None if the filing was already ingested.
def _process_filing(filing: dict, ticker: str, cik: str, filing_type: str) -> int | None:
    with get_connection() as conn:
        if filing_exists(conn, filing["accession"]):
            logger.info(f"Skipping {filing['accession']} — already ingested.")
            return None

        logger.info(f"Fetching {filing_type} {filing['period']} for {ticker}...")
        html = fetch_filing(filing["accession"], cik, filing["primary_doc"])

        sections = parse_sections(html, filing_type)
        logger.info(f"Parsed {len(sections)} sections.")

        filing_id = insert_filing(
            conn,
            ticker=ticker,
            cik=cik,
            filing_type=filing_type,
            period=filing["period"],
            accession=filing["accession"],
            filed_at=filing["filed_at"],
        )
        logger.info(f"Inserted filing id={filing_id}.")

        # Chunk all sections and track which section each chunk came from.
        all_chunks = []
        chunk_index = 0
        for section, text in sections.items():
            if not text or not text.strip():
                continue
            for chunk_text in _splitter.split_text(text):
                all_chunks.append({
                    "section": section,
                    "chunk_index": chunk_index,
                    "content": chunk_text,
                })
                chunk_index += 1

        logger.info(f"Split into {len(all_chunks)} chunks. Embedding...")

        # Embed all chunk contents in batched API calls.
        texts = [c["content"] for c in all_chunks]
        embeddings = embed_documents(texts)
        for chunk, embedding in zip(all_chunks, embeddings):
            chunk["embedding"] = embedding

        _insert_chunks(conn, filing_id, all_chunks)
        update_filing_status(conn, filing_id, "embedded")
        logger.info(f"Ingested {len(all_chunks)} chunks for filing id={filing_id}.")
        return filing_id


# Main entry point. Fetches and ingests all filings for a ticker and filing type.
# Returns the filing_ids that were newly ingested this run (already-ingested and failed filings are excluded).
def ingest(ticker: str, filing_type: str = "10-Q", limit: int = 5) -> list[int]:
    logger.info(f"Starting ingestion: {ticker} {filing_type} (limit={limit})")
    cik = get_cik(ticker)

    filings = get_10q_filings(cik, limit) if filing_type == "10-Q" else get_10k_filings(cik, limit)
    logger.info(f"Found {len(filings)} filings.")

    new_ids = []
    for filing in filings:
        try:
            filing_id = _process_filing(filing, ticker, cik, filing_type)
            if filing_id is not None:
                new_ids.append(filing_id)
        except Exception:
            logger.exception(f"Failed to ingest {filing['accession']}")
    return new_ids


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("ticker", help="Stock ticker, e.g. AAPL")
    parser.add_argument("--type", default="10-Q", choices=["10-Q", "10-K"])
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    ingest(args.ticker, args.type, args.limit)
