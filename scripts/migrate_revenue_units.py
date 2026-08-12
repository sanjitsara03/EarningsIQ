# One-off migration: rename signals.revenue -> revenue_usd and guidance_revenue ->
# guidance_revenue_usd, converting stored values from the old implicit-millions convention to raw
# USD, and rewriting segments JSONB entries from {revenue} (millions) to {revenue_usd} (raw USD).
# All conversions run ONLY when this run performed the rename, so re-running is a no-op.
#
# Usage: uv run python scripts/migrate_revenue_units.py
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.connection import get_connection

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MILLIONS = 1_000_000

RENAMES = [("revenue", "revenue_usd"), ("guidance_revenue", "guidance_revenue_usd")]


def _column_exists(cur, name: str) -> bool:
    cur.execute(
        "SELECT 1 FROM information_schema.columns WHERE table_name = 'signals' AND column_name = %s",
        (name,),
    )
    return cur.fetchone() is not None


# Rewrites one segments JSONB array from the old {revenue: millions} shape to {revenue_usd: raw}.
def _convert_segments(segments):
    out = []
    for seg in segments:
        seg = dict(seg)
        if "revenue" in seg:
            value = seg.pop("revenue")
            seg["revenue_usd"] = value * MILLIONS if value is not None else None
        out.append(seg)
    return out


def main() -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            renamed_now = []
            for old, new in RENAMES:
                if _column_exists(cur, old):
                    cur.execute(f"ALTER TABLE signals RENAME COLUMN {old} TO {new}")
                    renamed_now.append(new)
                    logger.info(f"Renamed signals.{old} -> {new}")
                else:
                    logger.info(f"signals.{old} already renamed — skipping its conversions")

            # Value conversions are tied to the rename happening in THIS run: after a completed
            # run the columns are already renamed, so a re-run converts nothing.
            for new in renamed_now:
                cur.execute(f"UPDATE signals SET {new} = {new} * %s WHERE {new} IS NOT NULL", (MILLIONS,))
                logger.info(f"Converted {cur.rowcount} {new} values from millions to raw USD")

            if "revenue_usd" in renamed_now:
                cur.execute("SELECT id, segments FROM signals WHERE segments IS NOT NULL")
                rows = cur.fetchall()
                for row_id, segments in rows:
                    cur.execute(
                        "UPDATE signals SET segments = %s WHERE id = %s",
                        (json.dumps(_convert_segments(segments)), row_id),
                    )
                logger.info(f"Converted segments JSONB for {len(rows)} rows")

            cur.execute("SELECT filing_id, revenue_usd FROM signals ORDER BY filing_id")
            for filing_id, rev in cur.fetchall():
                logger.info(f"filing_id={filing_id}: revenue_usd={float(rev):,.0f}" if rev else f"filing_id={filing_id}: revenue_usd=NULL")

    logger.info("Migration complete.")


if __name__ == "__main__":
    main()
