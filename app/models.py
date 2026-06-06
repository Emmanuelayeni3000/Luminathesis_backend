from pydantic import BaseModel, Field
from typing import Optional


# ─── Session Initialization ───────────────────────────────────────────────────

class InitSessionRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier (UUID recommended)")
    topic: str = Field(..., min_length=10, description="PhD research topic or thesis title")
    file_urls: list[str] = Field(
        default_factory=list,
        description="Vercel Blob URLs to PDF/DOCX files for parsing",
    )


class InitSessionResponse(BaseModel):
    session_id: str
    chunks_stored: int
    abstracts_harvested: int
    files_parsed: int
    message: str


# ─── Synthesis ────────────────────────────────────────────────────────────────

class SynthesizeRequest(BaseModel):
    session_id: str = Field(..., description="Session ID from initialize_session")
    topic: str = Field(..., min_length=10, description="PhD research topic")
    stream: bool = Field(default=True, description="Stream SSE output from Editor node")


class SynthesisResult(BaseModel):
    """Returned when stream=False"""
    session_id: str
    background: str
    problem_statement: str
    research_questions: str
    objectives: str
    hypotheses: str
    scope: str
    final_output: str


# ─── LangGraph State ──────────────────────────────────────────────────────────

class ThesisState(dict):
    """
    TypedDict-style state for the LangGraph pipeline.
    Using plain dict for LangGraph compatibility.
    """
    topic: str
    session_id: str
    context_chunks: list[str]
    background: str
    problem_statement: str
    research_questions: str
    objectives: str
    hypotheses: str
    scope: str
    final_output: str


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


# ─── Conversational Chat ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    message: str = Field(..., min_length=1, description="Conversational query")
    history: list[dict] = Field(default_factory=list, description="Chat history")
    user_name: Optional[str] = Field(None, description="The name of the logged-in user")


class ChatResponse(BaseModel):
    response: str
    sources: list[str] = []
    trigger_synthesis: bool = False
    synthesis_topic: Optional[str] = None
