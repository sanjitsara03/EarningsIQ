# Manages a psycopg2 connection pool backed by DATABASE_URL.
# The pool is lazily initialized on first use; forked RQ work-horses build their own pool.
import os
from contextlib import contextmanager

import psycopg2
import psycopg2.pool
from dotenv import load_dotenv

load_dotenv()

_pool = None


# Returns the pool, creating it on first call (after any fork).
def _get_pool():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=os.environ["DATABASE_URL"],
        )
    return _pool


@contextmanager
def get_connection():
    conn = _get_pool().getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _get_pool().putconn(conn)
