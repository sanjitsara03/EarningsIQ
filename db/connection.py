# Manages a psycopg2 connection pool backed by DATABASE_URL.
# Use get_connection() as a context manager
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.pool
from dotenv import load_dotenv

load_dotenv()

_pool = psycopg2.pool.SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    dsn=os.environ["DATABASE_URL"],
)


@contextmanager
def get_connection():
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)
