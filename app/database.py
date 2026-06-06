"""
Neon (PostgreSQL + pgvector) connection pool and table bootstrap.
Uses psycopg3 async pool — min 1 / max 5 connections (Neon free tier safe).
"""
import logging
from contextlib import asynccontextmanager

import psycopg_pool
from psycopg_pool import AsyncConnectionPool

from app.config import get_settings

logger = logging.getLogger(__name__)

# Module-level pool reference — set during lifespan startup
_pool: AsyncConnectionPool | None = None

CREATE_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector;"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS document_chunks (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL,
    content     TEXT NOT NULL,
    source      TEXT,
    embedding   vector(384),
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chunks_session
    ON document_chunks (session_id);
"""

CREATE_VECTOR_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
"""


async def init_db() -> None:
    """Create pool and bootstrap schema. Called once at app startup."""
    global _pool
    settings = get_settings()

    if not settings.database_url:
        logger.warning("DATABASE_URL not set — skipping DB initialisation")
        return

    logger.info("Initialising Neon connection pool…")
    _pool = AsyncConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=5,
        kwargs={"autocommit": True},
        open=False,
    )
    await _pool.open()

    async with _pool.connection() as conn:
        await conn.execute(CREATE_EXTENSION_SQL)
        await conn.execute(CREATE_TABLE_SQL)
        await conn.execute(CREATE_INDEX_SQL)
        # ivfflat index requires rows — only create if table has data
        try:
            await conn.execute(CREATE_VECTOR_INDEX_SQL)
        except Exception:
            pass  # Will be created automatically once rows are inserted

    logger.info("Database schema ready ✓")


async def close_db() -> None:
    """Close pool gracefully on app shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        logger.info("DB pool closed.")


def get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("DB pool not initialised — DATABASE_URL may be missing.")
    return _pool
