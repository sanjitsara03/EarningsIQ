#!/bin/bash
# Runs schema migration then starts the API server.
set -e

echo "Running schema migration..."
uv run python -c "
import os, psycopg2
conn = psycopg2.connect(os.environ['DATABASE_URL'])
conn.autocommit = True
cur = conn.cursor()
cur.execute(open('schema.sql').read())
conn.close()
print('Migration complete.')
"

echo "Starting server..."
exec uv run uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-8000}"
