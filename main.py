"""
LuminaThesis Backend — FastAPI Entry Point
==========================================
Architecture:
  - Lifespan: loads embedding model + DB pool once at startup
  - CORS: allows configured FRONTEND_URL
  - Routers: /api/initialize_session, /api/synthesize
  - Health: GET /health

Memory budget (Render 512 MB):
  - all-MiniLM-L6-v2 model  ≈  90 MB
  - FastAPI + dependencies   ≈  80 MB
  - Per-request headroom     ≈ 340 MB
"""
import os

# Must be set before torch (via sentence-transformers) is imported below —
# otherwise torch's OpenMP/MKL thread-pool init can hang indefinitely on
# Render's CPU-quota-limited instances, blocking startup before the app
# ever binds a port.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db, close_db
from app.services.embeddings import load_embedding_model, load_reranker
from app.routers import session, synthesis
from app.models import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: warm DB pool, embedding, and reranking models. Shutdown: close gracefully."""
    logger.info("━━━ LuminaThesis backend starting up ━━━")

    # Load embedding & re-ranking models into memory
    try:
        load_embedding_model()
        load_reranker()
    except Exception as e:
        logger.warning("Could not load models: %s", e)

    # Initialise Neon connection pool and bootstrap schema
    try:
        await init_db()
    except Exception as e:
        logger.warning("DB init failed (running without DB): %s", e)

    logger.info("━━━ Ready to serve requests ━━━")
    yield

    # Shutdown
    await close_db()
    logger.info("━━━ LuminaThesis backend shut down ━━━")


# ─── App ─────────────────────────────────────────────────────────────────────

settings = get_settings()

app = FastAPI(
    title="LuminaThesis API",
    description=(
        "Autonomous PhD research assistant backend. "
        "Harvests literature, indexes vectors, and runs a 3-node LangGraph "
        "pipeline (Extractor → Architect → Editor) to synthesise thesis content."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ─────────────────────────────────────────────────────────────────

app.include_router(session.router, prefix="/api", tags=["Session"])
app.include_router(synthesis.router, prefix="/api", tags=["Synthesis"])


# ─── Health ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    """Liveness probe for Render and uptime monitors."""
    return HealthResponse(status="ok")


# ─── Dev entrypoint ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
