"""
Multi-source academic literature harvester.
Queries Semantic Scholar and OpenAlex in parallel, deduplicates by DOI/title,
and returns up to `target_count` abstracts — widening source diversity for
comparison and reducing reliance on any single API.
Fully async — uses httpx with retries.
"""
import asyncio
import logging
import re
from itertools import zip_longest
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# `source` tag used when storing harvested literature abstracts in
# document_chunks, so chat RAG retrieval can exclude them when grounding
# answers about the user's own uploaded document(s).
HARVESTED_LITERATURE_SOURCE = "harvested_literature"

SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SCHOLAR_FIELDS = "title,abstract,year,authors,citationCount,externalIds,publicationTypes"

OPENALEX_SEARCH_URL = "https://api.openalex.org/works"
OPENALEX_HEADERS = {"User-Agent": "LuminaThesis/1.0 (https://github.com/Emmanuelayeni3000)"}


def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
    """OpenAlex stores abstracts as {word: [positions]} — rebuild plain text."""
    if not inverted_index:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted_index.items():
        for idx in idxs:
            positions.append((idx, word))
    positions.sort(key=lambda p: p[0])
    return " ".join(word for _, word in positions)


def _dedupe_key(paper: dict) -> str:
    doi = (paper.get("doi") or "").strip().lower()
    if doi:
        return doi.removeprefix("https://doi.org/")
    return paper.get("title", "").strip().lower()[:120]


def _merge_unique(*sources: list[dict], limit: int) -> list[dict]:
    """Interleave multiple result lists and drop duplicates by DOI/title."""
    merged: list[dict] = []
    seen: set[str] = set()
    for group in zip_longest(*sources):
        for paper in group:
            if paper is None:
                continue
            key = _dedupe_key(paper)
            if key in seen:
                continue
            seen.add(key)
            merged.append(paper)
            if len(merged) >= limit:
                return merged
    return merged


DOCUMENT_EXCERPT_CHARS = 4000

_QUERY_SYSTEM_PROMPT = (
    "You are a search query optimizer for academic literature databases. "
    "Return 3-6 core academic keywords describing the RESEARCH SUBJECT, separated by spaces. "
    "No quotes, punctuation, explanation or reasoning.\n"
    "The user's input may be a request about their work rather than a subject "
    "(e.g. 'recommend PhD topics from my BSc project', 'improve my chapter 2'). "
    "In that case take the subject from the DOCUMENT EXCERPT (title, abstract, objectives), "
    "never from words like thesis, project, BSc, PhD, topics or recommend."
)


def _clean_keywords(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"[\"'`*_#:,;.()\[\]]", " ", text)
    return " ".join(text.split()[:8])


async def build_search_query(topic: str, document_text: str = "") -> str:
    """Distill the user's topic (plus any uploaded document) into a keyword query."""
    settings = get_settings()
    content = f"USER INPUT:\n{topic}"
    if document_text.strip():
        content += f"\n\nDOCUMENT EXCERPT:\n{document_text[:DOCUMENT_EXCERPT_CHARS]}"
    try:
        from langchain_core.messages import SystemMessage, HumanMessage
        from app.services.llm import make_llm
        llm = make_llm(settings.editor_model, temperature=0.0)
        resp = await llm.ainvoke([SystemMessage(content=_QUERY_SYSTEM_PROMPT), HumanMessage(content=content)])
        query = _clean_keywords(resp.content)
        if query:
            return query
    except Exception as e:
        logger.warning("Keyword extraction failed, falling back to raw topic: %s", e)
    return topic[:100]


async def harvest_abstracts(topic: str, document_text: str = "") -> list[dict]:
    """
    Query Semantic Scholar and OpenAlex in parallel and return up to
    `settings.scholar_target_count` deduplicated papers with non-empty
    abstract text. `document_text` (the user's uploaded work) is used to find
    the research subject when `topic` is a request rather than a subject.
    """
    settings = get_settings()

    # 1. Distill conversational topic (+ uploaded document) into keywords
    clean_query = await build_search_query(topic, document_text)

    logger.info("Using clean query for literature search: '%s'", clean_query)

    target = settings.scholar_target_count

    results = await asyncio.gather(
        _harvest_semantic_scholar(clean_query, target, settings),
        _harvest_openalex(clean_query, target),
        return_exceptions=True,
    )

    semantic_papers: list[dict] = []
    openalex_papers: list[dict] = []
    for source_name, result in zip(("Semantic Scholar", "OpenAlex"), results):
        if isinstance(result, Exception):
            logger.error("%s harvest failed: %s", source_name, result)
        elif source_name == "Semantic Scholar":
            semantic_papers = result
        else:
            openalex_papers = result

    merged = _merge_unique(semantic_papers, openalex_papers, limit=target)

    logger.info(
        "Harvested %d abstracts (%d Semantic Scholar + %d OpenAlex, %d unique) for topic: %s",
        len(semantic_papers) + len(openalex_papers),
        len(semantic_papers),
        len(openalex_papers),
        len(merged),
        topic,
    )
    return merged


async def _harvest_semantic_scholar(clean_query: str, target: int, settings) -> list[dict]:
    batch = settings.scholar_batch_size

    headers: dict[str, str] = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    collected: list[dict] = []
    offset = 0

    async with httpx.AsyncClient(timeout=30.0) as client:
        while len(collected) < target:
            params = {
                "query": clean_query,
                "fields": SCHOLAR_FIELDS,
                "limit": batch,
                "offset": offset,
            }

            try:
                resp = await client.get(
                    SCHOLAR_SEARCH_URL, params=params, headers=headers
                )

                # Simple retry for rate limits
                if resp.status_code == 429:
                    logger.warning("Semantic Scholar 429 Rate Limit. Sleeping 2 seconds and retrying...")
                    await asyncio.sleep(2.0)
                    resp = await client.get(
                        SCHOLAR_SEARCH_URL, params=params, headers=headers
                    )

                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error("Semantic Scholar HTTP error: %s", e)
                break
            except httpx.RequestError as e:
                logger.error("Semantic Scholar request error: %s", e)
                break

            data = resp.json()
            papers = data.get("data", [])

            if not papers:
                break  # no more results

            for paper in papers:
                abstract = (paper.get("abstract") or "").strip()
                if not abstract:
                    continue

                authors = [
                    a.get("name", "") for a in (paper.get("authors") or [])
                ]
                external_ids = paper.get("externalIds") or {}

                collected.append(
                    {
                        "title": paper.get("title", "Unknown"),
                        "abstract": abstract,
                        "year": paper.get("year"),
                        "authors": authors,
                        "citation_count": paper.get("citationCount", 0),
                        "doi": external_ids.get("DOI", ""),
                        "source": "Semantic Scholar",
                    }
                )

                if len(collected) >= target:
                    break

            offset += batch

            # Respect rate limits — 1 req/s without API key, 10/s with
            await asyncio.sleep(0.15 if settings.semantic_scholar_api_key else 1.1)

    return collected[:target]


async def _harvest_openalex(clean_query: str, target: int) -> list[dict]:
    per_page = min(target, 100)
    collected: list[dict] = []
    page = 1
    max_pages = 3  # safety cap — avoids unbounded pagination on sparse queries

    async with httpx.AsyncClient(timeout=30.0) as client:
        while len(collected) < target and page <= max_pages:
            params = {
                "search": clean_query,
                "per_page": per_page,
                "page": page,
                "select": "title,abstract_inverted_index,publication_year,authorships,cited_by_count,doi",
            }

            try:
                resp = await client.get(OPENALEX_SEARCH_URL, params=params, headers=OPENALEX_HEADERS)
                resp.raise_for_status()
            except httpx.HTTPError as e:
                logger.error("OpenAlex request error: %s", e)
                break

            works = resp.json().get("results", [])
            if not works:
                break

            for work in works:
                abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
                if not abstract:
                    continue

                authors = [
                    (a.get("author") or {}).get("display_name", "")
                    for a in (work.get("authorships") or [])
                ]
                doi = (work.get("doi") or "").removeprefix("https://doi.org/")

                collected.append(
                    {
                        "title": work.get("title") or "Unknown",
                        "abstract": abstract,
                        "year": work.get("publication_year"),
                        "authors": authors,
                        "citation_count": work.get("cited_by_count", 0),
                        "doi": doi,
                        "source": "OpenAlex",
                    }
                )

                if len(collected) >= target:
                    break

            page += 1
            await asyncio.sleep(0.1)

    return collected[:target]


async def quick_search(query: str, limit: int = 5) -> list[dict]:
    """
    Lightweight, single-request lookup against Semantic Scholar and OpenAlex
    for live chat-time literature lookups. Unlike `harvest_abstracts`, this
    performs no keyword extraction or pagination — the caller is expected to
    supply an already-clean query (e.g. from the chat-intent classifier).
    """
    settings = get_settings()

    results = await asyncio.gather(
        _quick_search_semantic_scholar(query, limit, settings),
        _quick_search_openalex(query, limit),
        return_exceptions=True,
    )

    semantic_results = results[0] if isinstance(results[0], list) else []
    openalex_results = results[1] if isinstance(results[1], list) else []

    if isinstance(results[0], Exception):
        logger.warning("Semantic Scholar quick_search failed: %s", results[0])
    if isinstance(results[1], Exception):
        logger.warning("OpenAlex quick_search failed: %s", results[1])

    return _merge_unique(semantic_results, openalex_results, limit=limit)


async def _quick_search_semantic_scholar(query: str, limit: int, settings) -> list[dict]:
    headers: dict[str, str] = {}
    if settings.semantic_scholar_api_key:
        headers["x-api-key"] = settings.semantic_scholar_api_key

    params = {
        "query": query[:300],
        "fields": SCHOLAR_FIELDS,
        "limit": limit,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(SCHOLAR_SEARCH_URL, params=params, headers=headers)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("Semantic Scholar quick_search failed: %s", e)
        return []

    papers = resp.json().get("data", [])
    results = []
    for paper in papers:
        abstract = (paper.get("abstract") or "").strip()
        if not abstract:
            continue
        authors = [a.get("name", "") for a in (paper.get("authors") or [])]
        external_ids = paper.get("externalIds") or {}
        results.append(
            {
                "title": paper.get("title", "Unknown"),
                "abstract": abstract,
                "year": paper.get("year"),
                "authors": authors,
                "doi": external_ids.get("DOI", ""),
                "source": "Semantic Scholar",
            }
        )
        if len(results) >= limit:
            break

    return results


async def _quick_search_openalex(query: str, limit: int) -> list[dict]:
    params = {
        "search": query[:300],
        "per_page": limit,
        "select": "title,abstract_inverted_index,publication_year,authorships,doi",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(OPENALEX_SEARCH_URL, params=params, headers=OPENALEX_HEADERS)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("OpenAlex quick_search failed: %s", e)
        return []

    works = resp.json().get("results", [])
    results = []
    for work in works:
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        if not abstract:
            continue
        authors = [
            (a.get("author") or {}).get("display_name", "")
            for a in (work.get("authorships") or [])
        ]
        doi = (work.get("doi") or "").removeprefix("https://doi.org/")
        results.append(
            {
                "title": work.get("title") or "Unknown",
                "abstract": abstract,
                "year": work.get("publication_year"),
                "authors": authors,
                "doi": doi,
                "source": "OpenAlex",
            }
        )
        if len(results) >= limit:
            break

    return results


def format_abstracts_as_context(papers: list[dict]) -> str:
    """
    Format harvested papers into a numbered context block for the LLM.
    Produces: [N] Title (Year) — Authors\nAbstract text
    """
    lines = []
    for i, p in enumerate(papers, 1):
        author_str = (
            f"{p['authors'][0].split()[-1]}, {p['year']}"
            if p["authors"] and p["year"] and p["authors"][0].split()
            else str(p.get("year", "n.d."))
        )
        lines.append(
            f"[{i}] {p['title']} ({author_str})\n{p['abstract']}\n"
        )
    return "\n".join(lines)
