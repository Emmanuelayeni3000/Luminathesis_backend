"""
Central LLM factory for Groq.

Why this exists:
  - Model IDs are configuration, and Groq retires/renames models. A wrong ID
    used to surface only as a 500 in front of users. `verify_models()` checks
    the configured IDs against Groq's live model list at startup, and
    `resolve_model()` transparently swaps an unavailable ID for a working one.
  - Transient provider failures (model 404, 429, 5xx, connection errors) fall
    back to a second model instead of failing the request.
"""
import logging
from typing import Optional

import groq
import httpx
from langchain_groq import ChatGroq

from app.config import get_settings

logger = logging.getLogger(__name__)

GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"

# Preferred substitutes, best first, used when a configured model is missing
# and as the runtime fallback when the primary call fails.
FALLBACK_PREFERENCE = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b"]

# Errors worth retrying on another model. Deliberately excludes
# json_validate_failed (BadRequestError): the chat route repairs that itself.
FALLBACK_EXCEPTIONS = (
    groq.NotFoundError,
    groq.RateLimitError,
    groq.InternalServerError,
    groq.APIConnectionError,
    groq.APITimeoutError,
)

# None = not checked yet / list unavailable (then IDs are used as configured).
_available: Optional[set[str]] = None


async def verify_models() -> None:
    """Fetch Groq's model list and log every configured model that is missing."""
    global _available
    settings = get_settings()
    if not settings.groq_api_key:
        logger.error("GROQ_API_KEY is not set — all LLM calls will fail")
        return
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                GROQ_MODELS_URL,
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            )
            r.raise_for_status()
        _available = {m["id"] for m in r.json().get("data", [])}
    except Exception as e:
        logger.warning("Could not verify Groq models (%s) — using configured IDs as-is", e)
        return

    for role, model in (
        ("extractor", settings.extractor_model),
        ("architect", settings.architect_model),
        ("editor", settings.editor_model),
    ):
        if model in _available:
            logger.info("Model check ✓ %s → %s", role, model)
        else:
            logger.error(
                "Model check ✗ %s model '%s' is NOT available on Groq. Using '%s' instead. "
                "Fix EXTRACTOR_MODEL/ARCHITECT_MODEL/EDITOR_MODEL or app/config.py.",
                role, model, resolve_model(model),
            )


def resolve_model(model: str) -> str:
    """Return `model` if usable, else the first available preferred substitute."""
    if _available is None or model in _available:
        return model
    for candidate in FALLBACK_PREFERENCE:
        if candidate in _available:
            return candidate
    return model  # nothing better known; let the call fail loudly


def _fallback_for(primary: str) -> Optional[str]:
    """A different model to try if `primary` fails at call time."""
    for candidate in FALLBACK_PREFERENCE:
        if candidate != primary and (_available is None or candidate in _available):
            return candidate
    return None


def _chat(model: str, temperature: float, **kwargs) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=get_settings().groq_api_key,
        **kwargs,
    )


def make_llm(model: str, temperature: float, max_retries: int = 2):
    """Runnable for plain `.ainvoke(messages)` with model fallback."""
    primary_id = resolve_model(model)
    primary = _chat(primary_id, temperature, max_retries=max_retries)
    fb_id = _fallback_for(primary_id)
    if not fb_id:
        return primary
    return primary.with_fallbacks(
        [_chat(fb_id, temperature, max_retries=max_retries)],
        exceptions_to_handle=FALLBACK_EXCEPTIONS,
    )


def make_structured_llm(model: str, schema, temperature: float, method: str = "json_mode"):
    """Runnable that returns `schema` instances, with model fallback."""
    primary_id = resolve_model(model)
    primary = _chat(primary_id, temperature).with_structured_output(schema, method=method)
    fb_id = _fallback_for(primary_id)
    if not fb_id:
        return primary
    fallback = _chat(fb_id, temperature).with_structured_output(schema, method=method)
    return primary.with_fallbacks([fallback], exceptions_to_handle=FALLBACK_EXCEPTIONS)
