from app.services.graph import check_quality, finalize_node
import asyncio

CTX = ["Attention Is All You Need (Vaswani, 2017)\nTransformer."]


def state(text, loops=0):
    return {"final_output": text, "context_chunks": CTX, "loop_count": loops}


def test_clean_text_passes():
    assert check_quality(state("Transformers rely on attention (Vaswani, 2017).")) == "finalize"


def test_buzzword_loops_back_but_not_substrings():
    assert check_quality(state("This is crucial.")) == "editor"
    assert check_quality(state("The cruciality of it")) == "finalize"  # word boundary


def test_fabricated_citation_loops_back_with_feedback():
    s = state("Shown before (Smith, 2019).")
    assert check_quality(s) == "editor"
    assert "Smith, 2019" in s["feedback"]


def test_gives_up_after_max_loops_then_finalize_strips():
    s = state("Shown before (Smith, 2019).", loops=2)
    assert check_quality(s) == "finalize"
    out = asyncio.run(finalize_node(s))
    assert "Smith" not in out["final_output"]
    assert out["removed_citations"] == ["Smith, 2019"]
