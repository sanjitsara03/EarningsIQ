# One-off migration: re-embed all stored chunks with voyage-finance-2.
# Old OpenAI vectors and new Voyage vectors are not comparable, so every chunk must be
# re-embedded in one pass. Re-embeds from the chunk content already in the DB — no EDGAR
# refetch, no re-extraction, signals and risk scores are untouched.
#
# Usage: uv run python scripts/migrate_to_voyage.py
# Requires VOYAGE_API_KEY in the environment (.env is loaded). Safe to re-run.

import logging
import sys
import time
from pathlib import Path

# Allow running from anywhere without PYTHONPATH gymnastics.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import voyageai.error

from db.connection import get_connection
from pipeline.embeddings import EMBED_MODEL, embed_documents, embed_query

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 10 chunks ≈ 5K tokens — fits Voyage's no-payment-method tier (3 RPM / 10K TPM).
BATCH = 10


# Embed one batch, waiting out per-minute rate limits instead of dying.
def _embed_batch_throttled(texts: list[str]) -> list[list[float]]:
    while True:
        try:
            return embed_documents(texts)
        except voyageai.error.RateLimitError:
            logger.info("  rate limited — sleeping 25s (free tier is 3 requests/min)...")
            time.sleep(25)


def main() -> None:
    # Probe the model for its real dimension instead of trusting a hardcoded number.
    dim = len(embed_query("dimension probe"))
    logger.info(f"{EMBED_MODEL} produces {dim}-dim vectors.")

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Current column dimension (atttypmod holds the declared dim for vector columns).
            cur.execute("""
                SELECT atttypmod FROM pg_attribute
                WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
            """)
            current_dim = cur.fetchone()[0]
            logger.info(f"chunks.embedding is currently vector({current_dim}).")

            if current_dim != dim:
                logger.info(f"Altering column to vector({dim}) — existing vectors are wiped (incompatible anyway).")
                cur.execute("DROP INDEX IF EXISTS chunks_embedding_idx")
                cur.execute(f"ALTER TABLE chunks ALTER COLUMN embedding TYPE vector({dim}) USING NULL")

            # Only chunks still missing a vector — makes the script resumable after any interruption.
            cur.execute("SELECT id, content FROM chunks WHERE embedding IS NULL ORDER BY id")
            rows = cur.fetchall()
        logger.info(f"Re-embedding {len(rows)} chunks in batches of {BATCH}...")

        done = 0
        for i in range(0, len(rows), BATCH):
            batch = rows[i : i + BATCH]
            vectors = _embed_batch_throttled([content for _, content in batch])
            with conn.cursor() as cur:
                for (chunk_id, _), vec in zip(batch, vectors):
                    vec_str = "[" + ",".join(str(x) for x in vec) + "]"
                    cur.execute("UPDATE chunks SET embedding = %s::vector WHERE id = %s", (vec_str, chunk_id))
            conn.commit()  # persist each batch so an interruption never loses completed work
            done += len(batch)
            logger.info(f"  {done}/{len(rows)}")

        with conn.cursor() as cur:
            cur.execute("CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops)")
            cur.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NULL")
            nulls = cur.fetchone()[0]

    if nulls:
        raise SystemExit(f"ERROR: {nulls} chunks still have NULL embeddings — re-run the script.")
    logger.info("Migration complete: all chunks re-embedded, HNSW index rebuilt.")


if __name__ == "__main__":
    main()
