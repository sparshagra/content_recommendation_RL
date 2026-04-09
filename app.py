"""
Content Recommendation Environment — FastAPI Server
=====================================================
OpenEnv-compliant HTTP server for Hugging Face Spaces.

Entry point referenced in openenv.yaml:  entry_point: "app:app"
Docker CMD:  uvicorn app:app --host 0.0.0.0 --port 7860

Routes are split across three modules for clarity:
  routes/meta_routes.py   — /, /health, /metadata, /schema, /tasks
  routes/env_routes.py    — /reset, /step, /state, /catalog, /episode_summary
  routes/grader_routes.py — /grader, /graders, /grade, /grade/{task}, /grade/all

Shared state lives in env_state.py.
All Pydantic models live in models.py.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.env_routes import router as env_router
from routes.grader_routes import router as grader_router
from routes.meta_routes import router as meta_router

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Content Recommendation OpenEnv",
    description=(
        "Real-world RL environment for content recommendation. "
        "An agent recommends 5 items per step and receives reward signals "
        "based on click-through rate, content diversity, and long-term user retention. "
        "Implements the full OpenEnv spec: typed models, session-based grading, "
        "three difficulty tasks (easy / medium / hard)."
    ),
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(meta_router)
app.include_router(env_router)
app.include_router(grader_router)
