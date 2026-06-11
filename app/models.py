from pydantic import BaseModel, Field, field_validator
from typing import Optional


# ─── Session Initialization ───────────────────────────────────────────────────

class InitSessionRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier (UUID recommended)")
    topic: str = Field(..., min_length=1, description="PhD research topic or thesis title")
    file_urls: list[str] = Field(
        default_factory=list,
        description="Supabase Storage URLs to PDF/DOCX files for parsing",
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
    topic: str = Field(..., min_length=1, description="PhD research topic")
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
    response: str = Field(..., description="The main academic response or answer.")
    key_takeaways: list[str] = Field(..., description="3-5 concise bullet points summarizing the key takeaways.")
    suggested_followups: list[str] = Field(..., description="2-3 suggested follow-up questions the user can ask next.")
    sources: list[str] = Field(..., description="List of source document names referenced.")
    trigger_synthesis: bool = Field(..., description="True if the user explicitly requested to synthesize a thesis, otherwise false.")
    synthesis_topic: str = Field(..., description="The topic to synthesize if trigger_synthesis is true, otherwise an empty string.")

    @field_validator("key_takeaways", "suggested_followups", "sources", mode="before")
    @classmethod
    def _coerce_list_fields(cls, value):
        """
        The LLM occasionally returns a single string (one paragraph or a
        newline-separated block) instead of a JSON array for these fields.
        Coerce that into a list so it doesn't fail schema validation.
        """
        if value is None:
            return []
        if isinstance(value, str):
            lines = [line.strip(" \t-*•") for line in value.splitlines()]
            return [line for line in lines if line]
        return value


# ─── Export ──────────────────────────────────────────────────────────────────

class ExportRequest(BaseModel):
    topic: str = Field(..., description="PhD research topic or thesis title")
    background: Optional[str] = Field("", description="Background text")
    problem_statement: Optional[str] = Field("", description="Problem Statement text")
    research_questions: Optional[str] = Field("", description="Research Questions text")
    objectives: Optional[str] = Field("", description="Objectives text")
    hypotheses: Optional[str] = Field("", description="Hypotheses text")
    scope: Optional[str] = Field("", description="Scope text")
    final_output: Optional[str] = Field("", description="Final Prose text")

