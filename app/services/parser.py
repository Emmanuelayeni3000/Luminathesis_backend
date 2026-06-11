"""
LlamaParse cloud-side document parser.
Accepts Supabase Storage URLs — zero local file buffering (memory safe).
"""
import logging
from supabase import create_client, Client

from llama_parse import LlamaParse
from app.config import get_settings

logger = logging.getLogger(__name__)


def _get_parser() -> LlamaParse:
    settings = get_settings()
    return LlamaParse(
        api_key=settings.llama_cloud_api_key,
        result_type="markdown",
        verbose=False,
        language="en",
        # Instruct LlamaParse to preserve table structures and headings
        parsing_instruction=(
            "Extract all text content, preserving headings, paragraphs, "
            "and table data. Output clean markdown. Ignore page numbers, "
            "headers, footers, and decorative elements."
        ),
    )


def _get_supabase_client() -> Client | None:
    settings = get_settings()
    if not settings.supabase_url or not settings.supabase_service_role_key:
        logger.warning("Supabase credentials missing — Self-Destruct protocol disabled")
        return None
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


async def parse_documents(file_urls: list[str]) -> list[dict]:
    """
    Parse a list of remote file URLs via LlamaParse.

    Returns a list of dicts: {source_url, text}
    Files are processed cloud-side — never buffered locally.
    """
    if not file_urls:
        return []

    settings = get_settings()
    if not settings.llama_cloud_api_key:
        logger.warning("LLAMA_CLOUD_API_KEY not set — skipping document parsing")
        return []

    parser = _get_parser()
    results = []
    supabase_client = _get_supabase_client()

    for url in file_urls:
        filename = None
        # Extract filename from Supabase public URL
        # Supabase URL format: https://[project].supabase.co/storage/v1/object/public/research/[filename]
        if "/storage/v1/object/public/research/" in url:
            filename = url.split("/storage/v1/object/public/research/")[-1]

        try:
            logger.info("Parsing document: %s", url)
            # LlamaParse accepts URLs directly — no local download
            documents = await parser.aload_data(url)
            combined_text = "\n\n".join(doc.text for doc in documents if doc.text)
            results.append({"source_url": url, "text": combined_text})
            logger.info(
                "Parsed %d chars from %s", len(combined_text), url
            )
        except Exception as e:
            logger.error("Failed to parse %s: %s", url, e)
            # Non-fatal — continue with remaining files
            results.append({"source_url": url, "text": ""})
        finally:
            if filename and supabase_client:
                try:
                    logger.info("Self-Destruct: Deleting %s from Supabase Storage...", filename)
                    supabase_client.storage.from_("research").remove([filename])
                    logger.info("Self-Destruct complete for %s", filename)
                except Exception as ex:
                    logger.error("Self-Destruct failed for %s: %s", filename, ex)

    return results


async def upload_file_to_supabase(file_bytes: bytes, filename: str) -> str:
    """
    Upload file bytes directly to the 'research' bucket in Supabase Storage.
    Uses service role credentials to bypass Row-Level Security (RLS) constraints.
    Returns the public URL of the uploaded document.
    """
    settings = get_settings()
    supabase_client = _get_supabase_client()
    if not supabase_client:
        raise ValueError("Supabase is not configured on the backend (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY missing).")

    # Upload using storage API
    supabase_client.storage.from_("research").upload(
        path=filename,
        file=file_bytes,
        file_options={"cache-control": "3600", "upsert": "false"},
    )
    # Construct standard public URL
    public_url = f"{settings.supabase_url}/storage/v1/object/public/research/{filename}"
    logger.info("Uploaded document to Supabase storage: %s", public_url)
    return public_url
