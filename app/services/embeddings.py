"""
Embedding service using all-MiniLM-L6-v2 (≈90 MB, 384-dim).
Model is loaded once at startup and cached — never reloaded between requests.
Vectors are stored in Neon via psycopg3 COPY for throughput.
"""
import asyncio
import logging
from typing import TYPE_CHECKING

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.database import get_pool

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Module-level singletons — loaded once during app lifespan
_model: SentenceTransformer | None = None
_ranker = None


def load_reranker():
    """Load and cache the FlashRank re-ranking model."""
    global _ranker
    if _ranker is None:
        from flashrank import Ranker
        logger.info("Loading FlashRank re-ranking model…")
        _ranker = Ranker()
        logger.info("FlashRank re-ranking model loaded ✓")
    return _ranker


def load_embedding_model() -> SentenceTransformer:
    """Load and cache the embedding model. Call once at app startup."""
    global _model
    if _model is None:
        settings = get_settings()
        logger.info("Loading embedding model: %s", settings.embedding_model)
        _model = SentenceTransformer(settings.embedding_model)
        logger.info("Embedding model loaded ✓")
    return _model


def get_embedding_model() -> SentenceTransformer:
    if _model is None:
        return load_embedding_model()
    return _model


def chunk_text(
    text: str,
    source: str = "",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """
    Split text into overlapping chunks using RecursiveCharacterTextSplitter.
    Returns list of {content, source}.
    """
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [{"content": c, "source": source} for c in chunks if c.strip()]


async def store_chunks(session_id: str, chunks: list[dict]) -> int:
    """
    Embed and insert chunks into Neon document_chunks table.

    Args:
        session_id: Unique session identifier
        chunks: List of {content, source} dicts

    Returns:
        Number of chunks stored
    """
    if not chunks:
        return 0

    # We get pool inside the retry loop instead
    try:
        get_pool()
    except RuntimeError:
        logger.warning("DB pool unavailable — skipping vector storage")
        return 0

    model = get_embedding_model()
    texts = [c["content"] for c in chunks]

    logger.info("Embedding %d chunks…", len(texts))
    embeddings: np.ndarray = await asyncio.to_thread(
        model.encode,
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    rows = [
        (
            session_id,
            chunks[i]["content"],
            chunks[i].get("source", ""),
            embeddings[i].tolist(),
        )
        for i in range(len(chunks))
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            pool = get_pool()
            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.executemany(
                        """
                        INSERT INTO document_chunks (session_id, content, source, embedding)
                        VALUES (%s, %s, %s, %s::vector)
                        """,
                        rows,
                    )
            break
        except Exception as e:
            import asyncio
            logger.warning("DB insert failed on attempt %d: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                logger.error("Failed to store chunks after %d attempts", max_retries)
                raise
            await asyncio.sleep(1 + attempt)

    logger.info("Stored %d chunks for session %s", len(rows), session_id)
    return len(rows)


async def similarity_search(
    session_id: str, query: str, k: int | None = None
) -> list[str]:
    """
    Retrieve top-k most similar chunks to the query for a given session.
    Uses pgvector cosine distance operator `<=>` and FlashRank re-ranking.
    """
    settings = get_settings()
    top_k = k or settings.vector_top_k
    candidate_limit = min(top_k * 3, 40)

    try:
        pool = get_pool()
    except RuntimeError:
        logger.warning("DB pool unavailable — returning empty context")
        return []

    model = get_embedding_model()
    # CPU-bound: run off the event loop so other requests aren't blocked.
    query_embedding: np.ndarray = (
        await asyncio.to_thread(model.encode, [query], normalize_embeddings=True)
    )[0]

    max_retries = 3
    results = []
    for attempt in range(max_retries):
        try:
            async with pool.connection() as conn:
                rows = await conn.execute(
                    """
                    SELECT content
                    FROM document_chunks
                    WHERE session_id = %s
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (session_id, query_embedding.tolist(), candidate_limit),
                )
                results = await rows.fetchall()
            break
        except Exception as e:
            import asyncio
            logger.warning("DB similarity_search failed on attempt %d: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                logger.error("Failed to query database in similarity_search after %d attempts", max_retries)
                results = []
            else:
                await asyncio.sleep(0.5 * (attempt + 1))

    raw_texts = [r[0] for r in results]
    if not raw_texts:
        return []

    try:
        from flashrank import RerankRequest
        ranker = load_reranker()
        
        passages = [{"id": idx, "text": text} for idx, text in enumerate(raw_texts)]
        req = RerankRequest(query=query, passages=passages)
        reranked = await asyncio.to_thread(ranker.rerank, req)
        
        reranked_texts = [r["text"] for r in reranked[:top_k]]
        logger.info("FlashRank re-ranked %d candidates down to %d", len(raw_texts), len(reranked_texts))
        return reranked_texts
    except Exception as e:
        logger.error("FlashRank re-ranking failed: %s — falling back to raw vector search", e)
        return raw_texts[:top_k]


async def similarity_search_with_sources(
    session_id: str, query: str, k: int | None = None, exclude_sources: list[str] | None = None
) -> list[dict]:
    """
    Retrieve top-k chunks and their metadata sources for a given session.
    Uses pgvector cosine distance operator `<=>` and FlashRank re-ranking.
    Returns list of {"content": content_text, "source": source_metadata}.

    `exclude_sources` filters out chunks whose `source` column exactly
    matches one of the given values (e.g. harvested literature abstracts,
    which would otherwise compete with the user's own uploaded documents).
    """
    settings = get_settings()
    top_k = k or settings.vector_top_k
    candidate_limit = min(top_k * 3, 40)

    try:
        pool = get_pool()
    except RuntimeError:
        logger.warning("DB pool unavailable — returning empty context")
        return []

    model = get_embedding_model()
    # CPU-bound: run off the event loop so other requests aren't blocked.
    query_embedding: np.ndarray = (
        await asyncio.to_thread(model.encode, [query], normalize_embeddings=True)
    )[0]

    where_clause = "session_id = %s"
    params: list = [session_id]
    if exclude_sources:
        placeholders = ", ".join(["%s"] * len(exclude_sources))
        where_clause += f" AND source NOT IN ({placeholders})"
        params.extend(exclude_sources)
    params.append(query_embedding.tolist())
    params.append(candidate_limit)

    max_retries = 3
    results = []
    for attempt in range(max_retries):
        try:
            async with pool.connection() as conn:
                rows = await conn.execute(
                    f"""
                    SELECT content, source
                    FROM document_chunks
                    WHERE {where_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    params,
                )
                results = await rows.fetchall()
            break
        except Exception as e:
            import asyncio
            logger.warning("DB similarity_search_with_sources failed on attempt %d: %s", attempt + 1, e)
            if attempt == max_retries - 1:
                logger.error("Failed to query database in similarity_search_with_sources after %d attempts", max_retries)
                results = []
            else:
                await asyncio.sleep(0.5 * (attempt + 1))

    raw_candidates = [{"content": r[0], "source": r[1]} for r in results]
    if not raw_candidates:
        return []

    try:
        from flashrank import RerankRequest
        ranker = load_reranker()
        
        passages = [{"id": idx, "text": cand["content"]} for idx, cand in enumerate(raw_candidates)]
        req = RerankRequest(query=query, passages=passages)
        reranked = await asyncio.to_thread(ranker.rerank, req)
        
        reranked_results = []
        for r in reranked[:top_k]:
            idx = r["id"]
            reranked_results.append({
                "content": raw_candidates[idx]["content"],
                "source": raw_candidates[idx]["source"]
            })
        logger.info("FlashRank re-ranked %d candidates down to %d", len(raw_candidates), len(reranked_results))
        return reranked_results
    except Exception as e:
        logger.error("FlashRank re-ranking failed: %s — falling back to raw vector search", e)
        return raw_candidates[:top_k]
