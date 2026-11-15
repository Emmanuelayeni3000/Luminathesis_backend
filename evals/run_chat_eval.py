"""
End-to-end eval of POST /api/chat against a running backend.

    python -m evals.run_chat_eval --url https://ea-webcraft-luminathesis-backend.hf.space
    python -m evals.run_chat_eval --url http://localhost:8000 --session <id with uploaded docs>

Per-case checks: HTTP 200, schema fields, takeaways/follow-ups present, every
(Source: x) citation is in the returned `sources` (no ungrounded citations),
no system-prompt leakage, off-topic replies cite nothing. With --session, cases
marked needs_session_docs must also return at least one source.
Prints latency per case. Exits non-zero on any failure.
"""
import argparse
import json
import pathlib
import sys
import time
import uuid

import httpx

from app.services.citations import find_unsupported_citations
from app.services.scope import OUT_OF_SCOPE_MESSAGE

CASES = pathlib.Path(__file__).with_name("chat_cases.json")


def run(url: str, session: str | None) -> int:
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    failures = 0
    for c in cases:
        if c.get("needs_session_docs") and not session:
            print(f"SKIP  {c['id']} (pass --session with uploaded docs)")
            continue
        sid = session or str(uuid.uuid4())
        t0 = time.time()
        try:
            r = httpx.post(
                f"{url.rstrip('/')}/api/chat",
                json={"session_id": sid, "message": c["message"], "history": [], "user_name": "Eval"},
                timeout=120,
            )
        except Exception as e:
            print(f"FAIL  {c['id']}: request error {e}")
            failures += 1
            continue
        dt = time.time() - t0
        problems = []
        if r.status_code != 200:
            problems.append(f"HTTP {r.status_code}: {r.text[:120]}")
        else:
            d = r.json()
            if not d.get("response", "").strip():
                problems.append("empty response")
            if not c.get("off_topic") and c["id"] != "injection":
                if not d.get("key_takeaways"):
                    problems.append("no key_takeaways")
                if not d.get("suggested_followups"):
                    problems.append("no suggested_followups")
            bad = find_unsupported_citations(d.get("response", ""), known_sources=d.get("sources", []))
            if bad:
                problems.append(f"ungrounded citations: {bad}")
            if c.get("off_topic") and d.get("sources"):
                problems.append("off-topic reply cited sources")
            if c.get("off_topic") and d.get("response", "").strip() != OUT_OF_SCOPE_MESSAGE.strip():
                problems.append("off-topic prompt was not declined")
            if c.get("needs_session_docs") and not d.get("sources"):
                problems.append("expected sources from session docs, got none")
            for s in c.get("must_not_contain", []):
                if s.lower() in d.get("response", "").lower():
                    problems.append(f"leaked {s!r}")
        failures += bool(problems)
        print(f"{'FAIL' if problems else 'PASS'}  {c['id']:16} {dt:5.1f}s  {'; '.join(problems)}")
    print(f"\n{failures} failing case(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--session")
    a = ap.parse_args()
    sys.exit(run(a.url, a.session))
