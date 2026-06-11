"""
Semantic Scholar Graph API harvester.
Collects up to `target_count` abstracts with non-empty abstract text.
Fully async — uses httpx with retries.
"""
import asyncio
import logging
from typing import Optional

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

SCHOLAR_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SCHOLAR_FIELDS = "title,abstract,year,authors,citationCount,externalIds,publicationTypes"


async def harvest_abstracts(topic: str) -> list[dict]:
    """
    Query Semantic Scholar and return up to `settings.scholar_target_count`
    papers that have a non-empty abstract.
    """
    settings = get_settings()
    
    # 1. Distill conversational topic into keywords
    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = ChatGroq(model=settings.editor_model, temperature=0.0, api_key=settings.groq_api_key)
        sys_msg = "You are a search query optimizer. Extract 3-5 core academic keywords from the user's input. Return ONLY the keywords separated by spaces. Do not use quotes or punctuation."
        resp = await llm.ainvoke([SystemMessage(content=sys_msg), HumanMessage(content=topic)])
        clean_query = resp.content.strip()
    except Exception as e:
        logger.warning("Keyword extraction failed, falling back to raw topic: %s", e)
        clean_query = topic[:100]

    logger.info("Using clean query for Semantic Scholar: '%s'", clean_query)

    target = settings.scholar_target_count
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
                    }
                )

                if len(collected) >= target:
                    break

            offset += batch

            # Respect rate limits — 1 req/s without API key, 10/s with
            await asyncio.sleep(0.15 if settings.semantic_scholar_api_key else 1.1)

    logger.info("Harvested %d abstracts for topic: %s", len(collected), topic)
    return collected[:target]


async def quick_search(query: str, limit: int = 5) -> list[dict]:
    """
    Lightweight, single-request Semantic Scholar search for live chat-time
    literature lookups. Unlike `harvest_abstracts`, this performs no keyword
    extraction or pagination — the caller is expected to supply an
    already-clean query (e.g. from the chat-intent classifier).
    """
    settings = get_settings()
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
            if p["authors"] and p["year"]
            else str(p.get("year", "n.d."))
        )
        lines.append(
            f"[{i}] {p['title']} ({author_str})\n{p['abstract']}\n"
        )
    return "\n".join(lines)
