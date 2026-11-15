"""
POST /api/initialize_session

Pipeline:
  1. Parse uploaded documents via LlamaParse (URL-based, no local buffering)
  2. Harvest 50 Semantic Scholar abstracts for the topic
  3. Chunk all text with RecursiveCharacterTextSplitter
  4. Embed with all-MiniLM-L6-v2 and store in Neon pgvector
"""
import logging
import uuid

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.database import get_pool
from app.models import InitSessionRequest, InitSessionResponse
from app.services.parser import parse_documents, upload_file_to_supabase
from app.services.scholar import harvest_abstracts, format_abstracts_as_context, HARVESTED_LITERATURE_SOURCE
from app.services.embeddings import chunk_text, store_chunks
from app.services.ratelimit import check_rate_limit
from app.services.scope import OUT_OF_SCOPE_TOPIC_MESSAGE, is_academic_topic

logger = logging.getLogger(__name__)
router = APIRouter()

# Cooldown between repeated /initialize_session calls for the same session —
# tolerates the legitimate "attach more files mid-conversation" flow while
# blocking accidental rapid double-submits.
INIT_SESSION_COOLDOWN_SECONDS = 5.0


def _friendly_filename(url: str) -> str:
    """Strip the random UUID prefix from an uploaded file's Supabase URL for display."""
    basename = url.split("?")[0].rsplit("/", 1)[-1]
    parts = basename.split("_", 1)
    if len(parts) == 2 and len(parts[0]) == 36:
        return parts[1]
    return basename


async def _has_harvested_literature(session_id: str) -> bool:
    """True if literature abstracts were already harvested for this session."""
    try:
        pool = get_pool()
    except RuntimeError:
        return False

    try:
        async with pool.connection() as conn:
            cur = await conn.execute(
                "SELECT 1 FROM document_chunks WHERE session_id = %s AND source = %s LIMIT 1",
                (session_id, HARVESTED_LITERATURE_SOURCE),
            )
            return await cur.fetchone() is not None
    except Exception as e:
        logger.warning("Failed to check existing literature for session %s: %s", session_id, e)
        return False


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a file to Supabase storage via backend admin credentials (bypassing client RLS policies).
    Returns the public URL of the uploaded file.
    """
    try:
        import re
        file_bytes = await file.read()
        # Clean the original filename to remove special characters and spaces
        safe_orig_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', file.filename)
        random_filename = f"{uuid.uuid4()}_{safe_orig_name}"
        
        public_url = await upload_file_to_supabase(file_bytes, random_filename)
        return {"public_url": public_url}
    except Exception as e:
        logger.error("Failed to upload file to Supabase: %s", e)
        raise HTTPException(status_code=500, detail=f"File upload failed: {e}")


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

    wait = check_rate_limit(f"init:{body.session_id}", INIT_SESSION_COOLDOWN_SECONDS)
    if wait:
        raise HTTPException(
            status_code=429,
            detail=f"Please wait {wait}s before re-initializing this session.",
        )

    # Academic-scope gate: don't harvest literature or index anything for non-academic topics.
    if not await is_academic_topic(body.topic):
        raise HTTPException(status_code=422, detail=OUT_OF_SCOPE_TOPIC_MESSAGE)

    all_chunks: list[dict] = []
    files_parsed = 0
    files_failed: list[str] = []

    # ── Step 1: Parse uploaded documents ─────────────────────────────────────
    if body.file_urls:
        try:
            parsed_docs = await parse_documents(body.file_urls)
            for doc in parsed_docs:
                if doc["text"]:
                    chunks = chunk_text(doc["text"], source=doc["source_url"])
                    all_chunks.extend(chunks)
                    files_parsed += 1
                else:
                    files_failed.append(_friendly_filename(doc["source_url"]))
        except Exception as e:
            logger.error("Document parsing error: %s", e)
            # Non-fatal — continue without parsed files

    # ── Step 2: Harvest literature abstracts (Semantic Scholar + OpenAlex) ────
    # Only on the first initialization for this session — if the user attaches
    # more files later and initialize_session is called again, skip re-harvesting
    # so we don't burn extra Groq/OpenAlex quota or duplicate literature chunks.
    abstracts_harvested = 0
    already_harvested = await _has_harvested_literature(body.session_id)
    if already_harvested:
        logger.info("Literature already harvested for session %s — skipping re-harvest", body.session_id)
    else:
        try:
            papers = await harvest_abstracts(body.topic)
            abstracts_harvested = len(papers)

            scholar_text = format_abstracts_as_context(papers)
            if scholar_text:
                scholar_chunks = chunk_text(
                    scholar_text, source=HARVESTED_LITERATURE_SOURCE
                )
                all_chunks.extend(scholar_chunks)
        except Exception as e:
            logger.error("Literature harvest error: %s", e)
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

    literature_note = (
        "Using previously harvested literature."
        if already_harvested
        else f"{abstracts_harvested} abstracts harvested."
    )

    failed_note = ""
    if files_failed:
        failed_note = (
            f" Could not extract text from: {', '.join(files_failed)} "
            "(unsupported format or scanned/image-only file) — try a text-based PDF or DOCX."
        )

    return InitSessionResponse(
        session_id=body.session_id,
        chunks_stored=chunks_stored,
        abstracts_harvested=abstracts_harvested,
        files_parsed=files_parsed,
        message=(
            f"Session ready. {literature_note} "
            f"{files_parsed} document(s) indexed into {chunks_stored} new chunks.{failed_note}"
        ),
    )
