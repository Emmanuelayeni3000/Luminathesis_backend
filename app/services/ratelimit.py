"""
Minimal in-memory per-key cooldown tracker.

Single-process only (one Hugging Face Space container, no horizontal
scaling) — guards against accidental rapid-fire double-submits (e.g.
impatient double-clicks on "Generate") burning Groq/OpenAlex/LlamaParse
quota on duplicate work.
"""
import time

_last_call: dict[str, float] = {}


def check_rate_limit(key: str, cooldown_seconds: float) -> float:
    """
    Returns 0 if the call is allowed (and records it as the latest call for
    `key`), otherwise returns the number of seconds the caller must still wait.
    """
    now = time.monotonic()
    last = _last_call.get(key)
    if last is not None:
        elapsed = now - last
        if elapsed < cooldown_seconds:
            return round(cooldown_seconds - elapsed, 1)
    _last_call[key] = now
    return 0
