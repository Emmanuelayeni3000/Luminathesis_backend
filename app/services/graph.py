"""
LangGraph 3-node thesis synthesis pipeline.

Nodes:
  1. Extractor  → Llama 3.3  → Background + Problem Statement (with citations)
  2. Architect  → DeepSeek-R1 → Research Questions, Objectives, Hypotheses, Scope
  3. Editor     → Llama 3.3  → Unified, polished academic prose (buzzword-free)

State flows left to right; each node reads from and writes to ThesisState.
"""
import json
import logging
import re
from typing import AsyncIterator

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from app.config import get_settings
from app.prompts.extractor import EXTRACTOR_SYSTEM_PROMPT, EXTRACTOR_USER_TEMPLATE
from app.prompts.architect import ARCHITECT_SYSTEM_PROMPT, ARCHITECT_USER_TEMPLATE
from app.prompts.editor import EDITOR_SYSTEM_PROMPT, EDITOR_USER_TEMPLATE

logger = logging.getLogger(__name__)


# ─── LLM factory ─────────────────────────────────────────────────────────────

def _make_llm(model: str, temperature: float) -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=model,
        temperature=temperature,
        api_key=settings.groq_api_key,
        max_retries=3,
    )


def _parse_json_from_response(text: str) -> dict:
    """
    Extract the first JSON object from an LLM response.
    Handles DeepSeek-R1's <thinking>...</thinking> preamble.
    """
    # Strip <thinking> block if present
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()

    # Find JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        logger.warning("No JSON found in LLM response; returning raw text")
        return {"raw": text}

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.error("JSON parse error: %s\nRaw: %s", e, match.group()[:200])
        return {"raw": match.group()}


# ─── Node 1: Extractor ────────────────────────────────────────────────────────

async def extractor_node(state: dict) -> dict:
    """
    Llama 3.3 — Extracts Background and Problem Statement from context chunks.
    """
    settings = get_settings()
    llm = _make_llm(settings.extractor_model, temperature=0.3)

    context = "\n\n---\n\n".join(state.get("context_chunks", []))
    user_msg = EXTRACTOR_USER_TEMPLATE.format(
        topic=state["topic"],
        context=context or "(No additional documents provided — use general knowledge.)",
    )

    logger.info("[Extractor] Running Llama 3.3 extraction…")
    response = await llm.ainvoke(
        [
            SystemMessage(content=EXTRACTOR_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
    )

    parsed = _parse_json_from_response(response.content)
    logger.info("[Extractor] Done ✓")

    return {
        **state,
        "background": parsed.get("background", parsed.get("raw", "")),
        "problem_statement": parsed.get("problem_statement", ""),
    }


# ─── Node 2: Architect ───────────────────────────────────────────────────────

async def architect_node(state: dict) -> dict:
    """
    DeepSeek-R1 — Formulates Research Questions, Objectives, Hypotheses, Scope.
    Uses chain-of-thought reasoning (thinking block stripped from output).
    """
    settings = get_settings()
    llm = _make_llm(settings.architect_model, temperature=0.6)

    user_msg = ARCHITECT_USER_TEMPLATE.format(
        topic=state["topic"],
        background=state.get("background", ""),
        problem_statement=state.get("problem_statement", ""),
    )

    logger.info("[Architect] Running DeepSeek-R1 architecture…")
    response = await llm.ainvoke(
        [
            SystemMessage(content=ARCHITECT_SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ]
    )

    parsed = _parse_json_from_response(response.content)
    logger.info("[Architect] Done ✓")

    return {
        **state,
        "research_questions": parsed.get("research_questions", parsed.get("raw", "")),
        "objectives": parsed.get("objectives", ""),
        "hypotheses": parsed.get("hypotheses", ""),
        "scope": parsed.get("scope", ""),
    }


# ─── Node 3: Editor ──────────────────────────────────────────────────────────

async def editor_node(state: dict) -> dict:
    """
    Llama 3.3 — Rewrites all sections into unified, polished academic prose.
    Enforces buzzword blocklist and academic register.
    """
    settings = get_settings()
    llm = _make_llm(settings.editor_model, temperature=0.2)

    user_msg = EDITOR_USER_TEMPLATE.format(
        background=state.get("background", ""),
        problem_statement=state.get("problem_statement", ""),
        research_questions=state.get("research_questions", ""),
        objectives=state.get("objectives", ""),
        hypotheses=state.get("hypotheses", ""),
        scope=state.get("scope", ""),
    )

    messages = [
        SystemMessage(content=EDITOR_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]

    # If the Critic node generated correction feedback, append it to guide the LLM
    feedback = state.get("feedback")
    if feedback:
        logger.info("[Editor] Quality feedback received: %s", feedback)
        messages.append(
            HumanMessage(
                content=f"CRITIC FEEDBACK: {feedback}\n\nPlease revise the prose, ensuring all listed buzzwords are completely removed."
            )
        )

    logger.info("[Editor] Running Llama 3.3 editing pass…")
    response = await llm.ainvoke(messages)

    final = response.content.strip()
    if final.startswith("```markdown"):
        final = final[11:].strip()
    elif final.startswith("```"):
        final = final[3:].strip()
    if final.endswith("```"):
        final = final[:-3].strip()
        
    logger.info("[Editor] Done ✓ — %d chars", len(final))

    return {**state, "final_output": final}


# ─── Critic Router ───────────────────────────────────────────────────────────

def check_quality(state: dict) -> str:
    """
    Quality check conditional router.
    Evaluates final output for buzzwords and loops back to editor if needed.
    """
    final_text = state.get("final_output", "")
    if not final_text:
        return END

    # Blocklist terms
    buzzwords = ["delve", "tapestry", "underscores", "crucial", "testament", "pinnacle", "beacon", "groundbreaking"]
    found = [w for w in buzzwords if w in final_text.lower()]

    loop_count = state.get("loop_count", 0)

    if found and loop_count < 2:
        state["feedback"] = f"Blocklisted words found: {found}. Please rewrite the draft to remove these terms."
        state["loop_count"] = loop_count + 1
        logger.warning("[Critic] Quality check failed (Loop %d/2). Feedback: %s", state["loop_count"], state["feedback"])
        return "editor"

    if found:
        logger.warning("[Critic] Quality check failed but maximum loops reached. Proceeding.")
    else:
        logger.info("[Critic] Quality check passed ✓")

    return END


# ─── Graph Construction ───────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(dict)
    graph.add_node("extractor", extractor_node)
    graph.add_node("architect", architect_node)
    graph.add_node("editor", editor_node)

    graph.set_entry_point("extractor")
    graph.add_edge("extractor", "architect")
    graph.add_edge("architect", "editor")
    
    # Conditional loop routing
    graph.add_conditional_edges(
        "editor",
        check_quality,
        {
            "editor": "editor",
            END: END,
        }
    )

    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Compiled graph — built once at import time
thesis_graph = build_graph()


# ─── Public API ──────────────────────────────────────────────────────────────

async def run_synthesis(session_id: str, topic: str, context_chunks: list[str]) -> dict:
    """
    Run the full 3-node pipeline synchronously (for non-streaming responses).
    Returns the complete state dict.
    """
    initial_state = {
        "session_id": session_id,
        "topic": topic,
        "context_chunks": context_chunks,
        "background": "",
        "problem_statement": "",
        "research_questions": "",
        "objectives": "",
        "hypotheses": "",
        "scope": "",
        "final_output": "",
        "loop_count": 0,
        "feedback": "",
    }
    config = {"configurable": {"thread_id": session_id}}
    result = await thesis_graph.ainvoke(initial_state, config=config)
    return result


async def stream_synthesis(
    session_id: str, topic: str, context_chunks: list[str]
) -> AsyncIterator[str]:
    """
    Stream SSE-formatted events as each node completes.
    Yields strings formatted as `data: <json>\n\n`.
    """
    import json as _json

    state = {
        "session_id": session_id,
        "topic": topic,
        "context_chunks": context_chunks,
        "background": "",
        "problem_statement": "",
        "research_questions": "",
        "objectives": "",
        "hypotheses": "",
        "scope": "",
        "final_output": "",
        "loop_count": 0,
        "feedback": "",
    }

    # Node 1
    yield f"data: {_json.dumps({'event': 'node_start', 'node': 'extractor'})}\n\n"
    state = await extractor_node(state)
    yield f"data: {_json.dumps({'event': 'node_done', 'node': 'extractor', 'background': state['background'], 'problem_statement': state['problem_statement']})}\n\n"

    # Node 2
    yield f"data: {_json.dumps({'event': 'node_start', 'node': 'architect'})}\n\n"
    state = await architect_node(state)
    yield f"data: {_json.dumps({'event': 'node_done', 'node': 'architect', 'research_questions': state['research_questions'], 'objectives': state['objectives'], 'hypotheses': state['hypotheses'], 'scope': state['scope']})}\n\n"

    # Node 3 (Editor) with Critic Loop check
    state["loop_count"] = 0
    state["feedback"] = ""
    
    while True:
        yield f"data: {_json.dumps({'event': 'node_start', 'node': 'editor'})}\n\n"
        state = await editor_node(state)
        
        # Run Critic check
        next_step = check_quality(state)
        if next_step == "editor":
            # State properties like loop_count and feedback are updated inside check_quality
            yield f"data: {_json.dumps({'event': 'critic_failed', 'feedback': state['feedback'], 'loop_count': state['loop_count']})}\n\n"
        else:
            break

    yield f"data: {_json.dumps({'event': 'node_done', 'node': 'editor', 'final_output': state['final_output']})}\n\n"

    yield f"data: {_json.dumps({'event': 'complete', 'session_id': session_id})}\n\n"
