import app.services.llm as llm


def test_resolve_keeps_valid_model():
    llm._available = {"openai/gpt-oss-120b", "qwen/qwen3.8-27b"}
    assert llm.resolve_model("qwen/qwen3.8-27b") == "qwen/qwen3.8-27b"


def test_resolve_swaps_missing_model():
    llm._available = {"openai/gpt-oss-120b", "qwen/qwen3.8-27b"}
    assert llm.resolve_model("llama-3.3-70b-versatile") == "openai/gpt-oss-120b"


def test_resolve_passthrough_when_list_unknown():
    llm._available = None
    assert llm.resolve_model("whatever") == "whatever"


def test_fallback_is_a_different_model():
    llm._available = {"openai/gpt-oss-120b", "qwen/qwen3.8-27b"}
    assert llm._fallback_for("openai/gpt-oss-120b") == "qwen/qwen3.8-27b"
    assert llm._fallback_for("qwen/qwen3.8-27b") == "openai/gpt-oss-120b"
