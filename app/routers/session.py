"""
POST /api/initialize_session

Pipeline:
  1. Parse uploaded documents via LlamaParse (URL-based, no local buffering)
  2. Harvest 50 Semantic Scholar abstracts for the topic
  3. Chunk all text with RecursiveCharacterTextSplitter
  4. Embed with all-MiniLM-L6-v2 and store in Neon pgvector
"""
import logging

from fastapi import APIRouter, HTTPException

from app.models import InitSessionRequest, InitSessionResponse
from app.services.parser import parse_documents
from app.services.scholar import harvest_abstracts, format_abstracts_as_context
from app.services.embeddings import chunk_text, store_chunks

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/initialize_session", response_model=InitSessionResponse)
async def initialize_session(body: InitSessionRequest) -> InitSessionResponse:
    """
    Initialise a research session:
    - Parse any uploaded file URLs (LlamaParse)
    - Harvest 50 journal abstracts (Semantic Scholar)
    - Chunk, embed, and store all content in Neon (pgvector)
    """
    logger.info(
        "Initialising session %s | topic: %s | files: %d",
        body.session_id,
        body.topic[:60],
        len(body.file_urls),
    )

    all_chunks: list[dict] = []
    files_parsed = 0

    # ── Step 1: Parse uploaded documents ─────────────────────────────────────
    if body.file_urls:
        try:
            parsed_docs = await parse_documents(body.file_urls)
            for doc in parsed_docs:
                if doc["text"]:
                    chunks = chunk_text(doc["text"], source=doc["source_url"])
                    all_chunks.extend(chunks)
                    files_parsed += 1
        except Exception as e:
            logger.error("Document parsing error: %s", e)
            # Non-fatal — continue without parsed files

    # ── Step 2: Harvest Semantic Scholar abstracts ────────────────────────────
    abstracts_harvested = 0
    try:
        papers = await harvest_abstracts(body.topic)
        abstracts_harvested = len(papers)

        scholar_text = format_abstracts_as_context(papers)
        if scholar_text:
            scholar_chunks = chunk_text(
                scholar_text, source="semantic_scholar"
            )
            all_chunks.extend(scholar_chunks)
    except Exception as e:
        logger.error("Semantic Scholar error: %s", e)
        # Non-fatal — proceed with whatever we have

    # ── Step 3: Embed and store ───────────────────────────────────────────────
    chunks_stored = 0
    if all_chunks:
        try:
            chunks_stored = await store_chunks(body.session_id, all_chunks)
        except Exception as e:
            logger.error("Vector storage error: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to store document vectors: {e}",
            )

    return InitSessionResponse(
        session_id=body.session_id,
        chunks_stored=chunks_stored,
        abstracts_harvested=abstracts_harvested,
        files_parsed=files_parsed,
        message=(
            f"Session ready. {abstracts_harvested} abstracts and "
            f"{files_parsed} document(s) indexed into {chunks_stored} chunks."
        ),
    )
