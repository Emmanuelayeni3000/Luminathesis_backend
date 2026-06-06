# 🌌 LuminaThesis Backend

> **Autonomous PhD Literature Synthesis Flow & RAG Chat Advisor**
>
> *Built by [Ayeni Emmanuel](https://github.com/Emmanuelayeni3000)*


LuminaThesis is a high-performance Python FastAPI backend that orchestrates literature harvesting, vector embeddings, and a multi-agent LangGraph workflow to synthesize structured academic chapters. It also exposes a RAG-backed conversational PhD thesis advisor with strict anti-hallucination guardrails and metadata-driven citations.

---

## 🛠️ Architecture & Data Flow

```mermaid
graph TD
  A[Next.js Frontend] -->|1. Uploads PDF/Docs| B(Vercel Blob Storage)
  A -->|2. Topic + Doc URLs| C[FastAPI Server]
  C -->|3. Fetch Abstracts| D[Semantic Scholar API]
  C -->|4. Parse Documents| E[LlamaParse Cloud API]
  C -->|5. Chunk & Embed| F[sentence-transformers]
  C -->|6. Store Vectors| G[(Neon PostgreSQL + pgvector)]
  G -->|7. Semantic Query| H[FlashRank Re-ranking]
  H -->|8. Reranked Context| I[LangGraph Synthesis Flow]
  I -->|Step A: Extractor| J[Llama 3.3 70B]
  I -->|Step B: Architect| K[DeepSeek-R1 70B]
  I -->|Step C: Editor| L[Llama 3.3 70B]
  L -->|Step D: Critic Check| M{Quality Gate}
  M -->|Buzzwords Detected| L
  M -->|Pass| N[MemorySaver State Checkpoint]
  N -->|9. SSE Real-time Stream| A
```

---

## ✨ Features

- **🌐 Multi-Agent Synthesis (LangGraph)**:
  - **Extractor Node (Llama 3.3)**: Generates detailed Literature Review and Problem Statement blocks with inline citations.
  - **Architect Node (DeepSeek-R1)**: Leverages deep chain-of-thought thinking to draft structured Research Questions, SMART Objectives, and Methodology.
  - **Editor Node (Llama 3.3)**: Harmonizes tone and filters out overused AI buzzwords.
  - **Critic Loop**: Intercepts prose and triggers self-correction loops if buzzwords or quality standards are violated.
- **📚 Academic Harvester**: Queries Semantic Scholar to pull relevant paper abstracts based on the user's research topic.
- **⚡ Advanced RAG Pipeline**:
  - Embedding Model: `all-MiniLM-L6-v2` (384-dimensional).
  - Semantic Re-ranking: Uses `flashrank` (`ms-marco-TinyBERT-L-2-v2`) to prioritize the top relevant retrieved chunks.
  - Contextual Citations: Returns inline parenthetical citations dynamically mapped to original file sources and web URLs.
- **💬 Anti-Hallucination Chat Advisor**: Interactive session chat `/api/chat` with strict guardrails preventing the model from generating claims unsupported by retrieved context.
- **🔥 Server-Sent Events (SSE)**: Streams agent execution progress, critic logs, and final generated chapters to the client in real-time.

---

## 💻 Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Orchestration**: [LangGraph](https://langchain-ai.github.io/langgraph/) & [LangChain](https://python.langchain.com/)
- **Database**: [Neon Serverless PostgreSQL](https://neon.tech/) with `pgvector`
- **Vector Search**: Cosine Similarity + [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank)
- **Models**: Groq-hosted `llama-3.3-70b-versatile` & `deepseek-r1-distill-llama-70b`
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
│   │   ├── session.py          # Session endpoints (initialize, list, detail)
│   │   └── synthesis.py        # Core graph synthesis and chat advisor routes
│   ├── services/
│   │   ├── embeddings.py       # Embeddings generation, vector database & reranking
│   │   ├── graph.py            # LangGraph multi-agent synthesis state chart
│   │   ├── parser.py           # LlamaParse document parser
│   │   └── scholar.py          # Semantic Scholar literature search
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
| **POST** | `/api/initialize_session` | Bootstrap session and set up session-scoped vector partitions |
| **POST** | `/api/synthesize` | Trigger literature search, document ingestion, and LangGraph flow |
| **POST** | `/api/chat` | RAG-based interactive chat with user-personalized greeting and anti-hallucination |
| **GET** | `/api/sessions` | Retrieve all cached historical sessions |
| **GET** | `/api/sessions/{session_id}` | Retrieve specific session data (drafts, advisor logs, metadata) |
| **DELETE**| `/api/sessions/{session_id}` | Terminate and purge session metadata from database |

---

## ☁️ Deployment on Render

This repository is optimized for deployment via **Render Blueprints**.

1. Connect your Github Repository to your Render Dashboard.
2. Select **Blueprints** and Render will automatically detect `render.yaml`.
3. Provide the environment values for `GROQ_API_KEY`, `LLAMA_CLOUD_API_KEY`, and `DATABASE_URL`.
4. Click **Deploy**. The custom `build.sh` script will automatically compile using `uv` to speed up deployment times.

---

✨ Developed by [Ayeni Emmanuel](https://github.com/Emmanuelayeni3000)

