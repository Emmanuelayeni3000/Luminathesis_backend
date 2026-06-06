"""
LlamaParse cloud-side document parser.
Accepts Vercel Blob URLs — zero local file buffering (memory safe).
"""
import logging

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

    for url in file_urls:
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

    return results
