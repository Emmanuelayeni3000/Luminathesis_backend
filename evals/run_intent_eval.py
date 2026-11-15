"""
Golden-set eval for the intent classifier / query rewriter.

Run from the backend root (needs GROQ_API_KEY in .env or the environment):
    python -m evals.run_intent_eval

Exits non-zero if accuracy is below --min-accuracy (default 0.85).
"""
import argparse
import asyncio
import json
import pathlib
import sys

from app.services.intent import classify_chat_intent
from app.services.llm import verify_models

CASES = pathlib.Path(__file__).with_name("intent_cases.json")


async def main(min_accuracy: float) -> int:
    await verify_models()
    cases = json.loads(CASES.read_text(encoding="utf-8"))
    passed = 0
    for c in cases:
        got = await classify_chat_intent(c["message"], c["history"])
        ok = True
        if "expect" in c:
            ok &= got["intent"] == c["expect"]
        if "expect_not" in c:
            ok &= got["intent"] != c["expect_not"]
        for word in c.get("search_query_must_contain", []):
            ok &= word.lower() in got["search_query"].lower()
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'}  {c['message'][:60]!r:64} -> {got['intent']} | q={got['search_query'][:50]!r}")
    acc = passed / len(cases)
    print(f"\nintent accuracy: {passed}/{len(cases)} = {acc:.0%}")
    return 0 if acc >= min_accuracy else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-accuracy", type=float, default=0.85)
    sys.exit(asyncio.run(main(ap.parse_args().min_accuracy)))
