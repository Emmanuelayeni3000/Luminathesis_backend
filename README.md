---
title: LuminaThesis Backend
emoji: 🌌
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🌌 LuminaThesis Backend

> **Autonomous PhD Literature Synthesis Flow & RAG Chat Advisor**
>
> *Built by [Ayeni Emmanuel](https://github.com/Emmanuelayeni3000)*


LuminaThesis is a high-performance Python FastAPI backend that orchestrates literature harvesting, vector embeddings, and a multi-agent LangGraph workflow to synthesize structured academic chapters. It also exposes a RAG-backed conversational PhD thesis advisor with strict anti-hallucination guardrails and metadata-driven citations.

---

## 🛠️ Architecture & Data Flow

```mermaid
graph TD
  A[Next.js Frontend] -->|1. Uploads PDF/Docs| B(Supabase Storage)
  A -->|2. Topic + Doc URLs| C[FastAPI Server]
  C -->|3. Fetch Abstracts| D[Semantic Scholar + OpenAlex APIs]
  C -->|4. Parse Documents| E[LlamaParse Cloud API]
  C -->|5. Chunk & Embed| F[sentence-transformers]
  C -->|6. Store Vectors| G[(Neon PostgreSQL + pgvector)]
  G -->|7. Semantic Query| H[FlashRank Re-ranking]
  H -->|8. Reranked Context| I[LangGraph Synthesis Flow]
  I -->|Step A: Extractor| J[Llama 3.3 70B]
  I -->|Step B: Architect| K[Qwen3 32B]
  I -->|Step C: Editor| L[Llama 3.3 70B]
  L -->|Step D: Critic Check| M{Quality Gate}
  M -->|Buzzwords Detected| L
  M -->|Pass| N[MemorySaver State Checkpoint]
  N -->|9. SSE Real-time Stream| A
```

### 💬 Chat Advisor Routing

Every `/api/chat` turn is first triaged by a lightweight intent classifier (`app/services/intent.py`) before any LLM generation happens:

```mermaid
graph TD
  A[User Message + History] --> B[classify_chat_intent]
  B -->|thesis_synthesis| C[Trigger LangGraph Synthesis Pipeline]
  B -->|literature_search| D[Vector Search + Live quick_search across Semantic Scholar & OpenAlex]
  B -->|general_chat| E[Vector Search Only]
  B -->|off_topic| F[Skip Retrieval]
  D --> G[Editor LLM — json_mode structured output]
  E --> G
  F --> G
  G -->|ChatResponse| A
```

- **Intent classification & query rewriting**: a single fast call resolves pronouns/follow-ups ("what about its limitations?") into standalone retrieval queries and routes the turn to one of `thesis_synthesis`, `literature_search`, `general_chat`, or `off_topic`. The synthesis pipeline only fires on an explicit, unambiguous request — outline/table-of-contents/brainstorming requests stay conversational.
- **Live literature lookups**: `literature_search` turns are augmented with a real-time `quick_search` across Semantic Scholar and OpenAlex, deduplicated by DOI/title, alongside the user's indexed documents.
- **Resilient structured output**: chat responses are generated with Groq `json_mode` (instead of strict tool-calling) plus Pydantic coercion validators, so malformed `key_takeaways`/`suggested_followups` fields no longer crash the endpoint.

---

## ✨ Features

- **🌐 Multi-Agent Synthesis (LangGraph)**:
  - **Extractor Node (Llama 3.3)**: Generates detailed Literature Review and Problem Statement blocks with inline citations.
  - **Architect Node (Qwen3 32B)**: Leverages deep chain-of-thought thinking to draft structured Research Questions, SMART Objectives, and Methodology.
  - **Editor Node (Llama 3.3)**: Harmonizes tone and filters out overused AI buzzwords.
  - **Critic Loop**: Intercepts prose and triggers self-correction loops if buzzwords or quality standards are violated.
- **📚 Academic Harvester**: Queries Semantic Scholar and OpenAlex in parallel, deduplicated by DOI/title, to pull relevant paper abstracts from multiple sources based on the user's research topic.
- **⚡ Advanced RAG Pipeline**:
  - Embedding Model: `all-MiniLM-L6-v2` (384-dimensional).
  - Semantic Re-ranking: Uses `flashrank` (`ms-marco-TinyBERT-L-2-v2`) to prioritize the top relevant retrieved chunks.
  - Contextual Citations: Returns inline parenthetical citations dynamically mapped to original file sources and web URLs.
- **🧭 Smart Chat Intent Routing**: Every chat turn is classified (`thesis_synthesis` / `literature_search` / `general_chat` / `off_topic`) and the latest message is rewritten into a standalone, history-aware retrieval query before vector search runs.
- **🔎 Live Literature Lookups in Chat**: `literature_search` turns trigger a real-time `quick_search` across Semantic Scholar and OpenAlex, blending fresh, deduplicated paper results with the user's indexed documents.
- **💬 Anti-Hallucination Chat Advisor**: Interactive session chat `/api/chat` with strict guardrails preventing the model from generating claims unsupported by retrieved context, resilient `json_mode` structured output, and edge-case handling for empty context, overlong pastes, and table-of-contents formatting.
- **📄 Chapter Export**: Export the synthesized chapter to styled `.docx` or `.pdf` via `/api/export/docx` and `/api/export/pdf`.
- **🔥 Server-Sent Events (SSE)**: Streams agent execution progress, critic logs, and final generated chapters to the client in real-time.

---

## 💻 Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) & [LangChain](https://python.langchain.com/)
- **Database**: [Neon Serverless PostgreSQL](https://neon.tech/) with `pgvector`
- **Vector Search**: Cosine Similarity + [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)
- **Models**: Groq-hosted `llama-3.3-70b-versatile` & `qwen/qwen3-32b`
- **Package Manager**: [Astral uv](https://github.com/astral-sh/uv)

---

## 🚀 Getting Started

### Prerequisites

Ensure you have **Python 3.14+** and **uv** installed.

```bash
# Verify uv installation
uv --version
```

### Installation

1. Clone the repository and navigate to the backend directory:
   ```bash
   cd luminathesis-backend
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv pip sync requirements.txt
   ```

3. Create your `.env` configuration file from the template:
   ```env
   # API Keys
   GROQ_API_KEY=your-groq-key
   LLAMA_CLOUD_API_KEY=your-llamacloud-key
   SEMANTIC_SCHOLAR_API_KEY=your-optional-scholar-key

   # Supabase Storage (used for file uploads via /api/upload)
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key

   # Database Connection
   DATABASE_URL=postgresql://user:pass@ep-host.us-east-1.aws.neon.tech/neondb?sslmode=require

   # CORS Configuration
   FRONTEND_URL=http://localhost:3000
   ```

### Running Locally

To spin up the local development server with hot reload:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server will be live at `http://localhost:8000`. You can inspect the interactive OpenAPI documentation at `http://localhost:8000/docs`.

---

## 🗂️ Project Structure

```bash
luminathesis-backend/
├── app/
│   ├── routers/
│   │   ├── session.py          # Upload + session initialization endpoints
│   │   └── synthesis.py        # Synthesis pipeline, chat advisor, and export routes
│   ├── services/
│   │   ├── embeddings.py       # Embeddings generation, vector database & reranking
│   │   ├── graph.py            # LangGraph multi-agent synthesis state chart
│   │   ├── intent.py           # Chat intent classification & query rewriting
│   │   ├── parser.py           # LlamaParse document parser + Supabase upload
│   │   ├── scholar.py          # Semantic Scholar literature harvesting & live search
│   │   └── document_generator.py # DOCX/PDF chapter export rendering
│   ├── prompts/                # System prompts for the Extractor/Architect/Editor nodes
│   ├── config.py               # Settings and env variables loader
│   ├── database.py             # Neon connection pool and bootstrap
│   └── models.py               # Pydantic schemas (requests, responses, health)
├── main.py                     # Entrypoint with lifespan pre-warming
├── pyproject.toml              # Dependencies and project specifications
├── requirements.txt            # Frozen, compiled dependencies list
├── build.sh                    # Deployment build script for Render
└── render.yaml                 # Deployment blueprint config
```

---

## 🔌 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| **GET** | `/health` | Server liveness check |
| **POST** | `/api/upload` | Upload a file to Supabase Storage and return its public URL |
| **POST** | `/api/initialize_session` | Parse uploaded docs, harvest abstracts, and store embeddings for a session |
| **POST** | `/api/synthesize` | Run the LangGraph synthesis flow (SSE stream or JSON) for a topic |
| **POST** | `/api/chat` | Intent-routed RAG chat advisor with anti-hallucination guardrails |
| **POST** | `/api/export/docx` | Export the synthesized thesis chapter as a styled `.docx` |
| **POST** | `/api/export/pdf` | Export the synthesized thesis chapter as a styled `.pdf` |

---

## ☁️ Deployment on Render

This repository is optimized for deployment via **Render Blueprints**.

1. Connect your Github Repository to your Render Dashboard.
2. Select **Blueprints** and Render will automatically detect `render.yaml`.
3. Provide the environment values for `GROQ_API_KEY`, `LLAMA_CLOUD_API_KEY`, and `DATABASE_URL`.
4. Click **Deploy**. The custom `build.sh` script will automatically compile using `uv` to speed up deployment times.

---

✨ Developed by [Ayeni Emmanuel](https://github.com/Emmanuelayeni3000)

