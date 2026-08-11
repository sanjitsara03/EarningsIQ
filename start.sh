#!/bin/bash
# Runs schema migration then starts the API server.
set -e

echo "Running schema migration..."
uv run python -c "
import os, psycopg2

conn = psycopg2.connect(os.environ['DATABASE_URL'])
conn.autocommit = True
cur = conn.cursor()

cur.execute(\"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'filings');\")
exists = cur.fetchone()[0]

if not exists:
    cur.execute(open('schema.sql').read())
    print('Migration complete.')
else:
    print('Tables already exist, skipping.')

conn.close()
"

echo "Running column migrations..."
uv run python scripts/migrate_revenue_units.py

echo "Starting server..."
exec uv run uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
