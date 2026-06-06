"""
POST /api/synthesize

Triggers the LangGraph 3-node pipeline:
  Extractor → Architect → Editor

Supports both:
  - stream=true  → Server-Sent Events (text/event-stream)
  - stream=false → JSON response with full SynthesisResult
"""
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models import SynthesizeRequest, SynthesisResult, ChatRequest, ChatResponse
from app.services.embeddings import similarity_search, similarity_search_with_sources
from app.services.graph import run_synthesis, stream_synthesis
from app.config import get_settings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/synthesize")
async def synthesize(body: SynthesizeRequest):
    """
    Run the LangGraph synthesis pipeline for a session.

    Retrieves top-k context chunks from Neon, then runs:
      Extractor (Llama 3.3) → Architect (DeepSeek-R1) → Editor (Llama 3.3)

    Returns SSE stream if `stream=true`, else JSON.
    """
    logger.info(
        "Synthesis request | session: %s | topic: %s | stream: %s",
        body.session_id,
        body.topic[:60],
        body.stream,
    )

    # ── Retrieve relevant context from Neon ───────────────────────────────────
    try:
        context_chunks = await similarity_search(
            session_id=body.session_id,
            query=body.topic,
        )
        logger.info("Retrieved %d context chunks", len(context_chunks))
    except Exception as e:
        logger.warning("Context retrieval failed: %s — proceeding without context", e)
        context_chunks = []

    # ── Streaming response ────────────────────────────────────────────────────
    if body.stream:
        async def event_generator():
            try:
                async for chunk in stream_synthesis(
                    session_id=body.session_id,
                    topic=body.topic,
                    context_chunks=context_chunks,
                ):
                    yield chunk
            except Exception as e:
                logger.error("Stream error: %s", e)
                yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    # ── Non-streaming JSON response ───────────────────────────────────────────
    try:
        result = await run_synthesis(
            session_id=body.session_id,
            topic=body.topic,
            context_chunks=context_chunks,
        )
    except Exception as e:
        logger.error("Synthesis pipeline error: %s", e)
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    return SynthesisResult(
        session_id=body.session_id,
        background=result.get("background", ""),
        problem_statement=result.get("problem_statement", ""),
        research_questions=result.get("research_questions", ""),
        objectives=result.get("objectives", ""),
        hypotheses=result.get("hypotheses", ""),
        scope=result.get("scope", ""),
        final_output=result.get("final_output", ""),
    )


async def check_thesis_intent(message: str, history: list[dict]) -> tuple[bool, str | None]:
    """
    Classify whether the user's query expresses an intent to start thesis synthesis/research.
    Returns (is_thesis_intent, extracted_topic).
    """
    settings = get_settings()
    llm = ChatGroq(
        model=settings.editor_model,
        temperature=0.0,
        api_key=settings.groq_api_key,
    )
    
    # Format recent history for the classifier context
    history_context = []
    for h in history[-4:]:
        role = h.get("role")
        content = h.get("content", "")
        history_context.append(f"{role.upper()}: {content}")
    history_str = "\n".join(history_context)
    
    system_prompt = (
        "You are an academic intent classifier. Your task is to analyze the user's latest message "
        "and determine if they are proposing, requesting, or starting a thesis/dissertation research synthesis session on a specific topic.\n\n"
        "Thesis synthesis intent includes requests to:\n"
        "- Research a specific thesis topic (e.g. 'I want to write about X', 'Let's do research on Y').\n"
        "- Run the synthesis/generation pipeline on an academic area.\n"
        "- Brainstorming a specific topic that they clearly want to research right now.\n\n"
        "Non-thesis intent includes:\n"
        "- General greetings (e.g. 'hi', 'hello').\n"
        "- Generic conceptual questions (e.g. 'what is a literature review?', 'how do I structure a thesis?').\n"
        "- General conversation or questions unrelated to a specific topic they want to research right now.\n\n"
        f"Conversation History:\n{history_str}\n\n"
        f"Latest User Message: \"{message}\"\n\n"
        "You MUST respond ONLY with a JSON object in the following format. Do not write any other explanation or text:\n"
        "{\n"
        "  \"is_thesis_intent\": true or false,\n"
        "  \"extracted_topic\": \"A clean, formal, concise thesis topic/title string of 10+ characters (only if is_thesis_intent is true, otherwise null)\"\n"
        "}"
    )
    
    try:
        response = await llm.ainvoke([SystemMessage(content=system_prompt)])
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        import json
        data = json.loads(content)
        is_intent = data.get("is_thesis_intent", False)
        topic = data.get("extracted_topic")
        
        if is_intent and topic and len(topic.strip()) >= 10:
            return True, topic.strip()
        return False, None
    except Exception as e:
        logger.error("Thesis intent classification failed: %s", e)
        return False, None


@router.post("/chat", response_model=ChatResponse)
async def chat_advisor(body: ChatRequest) -> ChatResponse:
    """
    RAG-backed conversational advisor. Answers Emily's questions about her thesis
    or general academic queries, preventing hallucinations using vector DB context
    and generating precise source citations/references.
    """
    logger.info("Chat advisor request | session: %s | msg: %s", body.session_id, body.message[:60])
    
    # ── Check for thesis synthesis intent first ───────────────────────────────
    is_thesis, topic = await check_thesis_intent(body.message, body.history)
    if is_thesis and topic:
        logger.info("Detected thesis intent for topic: %s", topic)
        return ChatResponse(
            response=(
                f"Excellent topic! I've identified your thesis area: **\"{topic}\"**.\n\n"
                "I am triggering the LuminaThesis synthesis pipeline right now to harvest academic literature "
                "from Semantic Scholar and structure your chapter. Sit tight!"
            ),
            sources=[],
            trigger_synthesis=True,
            synthesis_topic=topic
        )

    # 1. Retrieve context chunks with their metadata sources
    try:
        context_items = await similarity_search_with_sources(
            session_id=body.session_id,
            query=body.message,
            k=6
        )
        logger.info("Retrieved %d context chunks for chat advisor", len(context_items))
    except Exception as e:
        logger.error("Chat context retrieval failed: %s", e)
        context_items = []

    # 2. Setup prompt and guidelines for RAG + Anti-hallucination + References
    context_text_blocks = []
    unique_sources = set()
    for item in context_items:
        source_name = item.get("source", "Unknown Document")
        unique_sources.add(source_name)
        context_text_blocks.append(f"Source: {source_name}\nContent: {item.get('content')}")
        
    context_text = "\n\n---\n\n".join(context_text_blocks)
    
    user_name = body.user_name or "Emily"
    system_prompt = (
        f"You are LuminaThesis Advisor, a helpful academic assistant guiding a PhD student ({user_name}) through their thesis chapter.\n"
        f"Your goal is to answer their questions with absolute academic integrity, avoiding any hallucinations.\n\n"
        "CRITICAL INSTRUCTIONS FOR ANTI-HALLUCINATION & CITATIONS:\n"
        f"1. If {user_name}'s question is about their uploaded files or their specific draft, you MUST base your answer strictly on the provided 'RETRIEVED CONTEXT' chunks below.\n"
        "2. If you use information from a context chunk, you MUST cite its 'Source' inline parenthetically, for example: (Source: remote_learning_trends.pdf) or (Source: semantic_scholar).\n"
        f"3. If the answer cannot be found in the RETRIEVED CONTEXT, state clearly: 'I cannot find that information in the uploaded research documents or draft chunks.' - Do not make up or hallucinate details.\n"
        "4. If you used any RETRIEVED CONTEXT to answer their question, you MUST append a References section at the very end of your response formatted exactly like this:\n\n"
        "## References\n"
        "- **[Source Name/URL/Filename]** (Inline Citation Name)\n\n"
        f"5. If their question is a general academic query (e.g. general definitions, formatting guides, Python/LaTeX help) and doesn't refer to the uploaded texts, you may answer using your general knowledge but clarify that it is general advice (and do not add a References section).\n"
        "6. Always maintain a professional, encouraging academic tone.\n\n"
        f"RETRIEVED CONTEXT CHUNKS:\n{context_text or '(No uploaded research context found for this query)'}"
    )

    # 3. Build chat history messages
    messages = [SystemMessage(content=system_prompt)]
    for h in body.history[-8:]:
        role = h.get("role")
        content = h.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))
            
    # Add current user message
    messages.append(HumanMessage(content=body.message))

    # 4. Invoke LLM (Llama 3.3)
    try:
        settings = get_settings()
        llm = ChatGroq(
            model=settings.editor_model,
            temperature=0.4,
            api_key=settings.groq_api_key,
        )
        response = await llm.ainvoke(messages)
        
        return ChatResponse(
            response=response.content,
            sources=list(unique_sources)
        )
    except Exception as e:
        logger.error("Chat advisor LLM invocation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Advisor service error: {e}")
