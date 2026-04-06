"""
Content Recommendation Environment — FastAPI Server
=====================================================
Exposes the OpenEnv-compliant environment over HTTP for Hugging Face Spaces.

Endpoints:
  GET  /health      — liveness check (returns HTTP 200)
  POST /reset       — reset environment, returns initial observation
  POST /step        — execute one step, returns StepResult
  GET  /state       — get current user state

Usage (local):
  uvicorn app:app --host 0.0.0.0 --port 7860

Usage (Docker / HF Spaces):
  Dockerfile CMD triggers this automatically.
"""

import os
import asyncio
from dataclasses import asdict
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from content_rec_env import (
    ContentRecEnv,
    RecommendationAction,
    TaskDifficulty,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Content Recommendation OpenEnv",
    description=(
        "A real-world RL environment for content recommendation. "
        "Agents learn to recommend articles/videos/songs to maximise "
        "engagement, diversity, and long-term user retention."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Environment state (one global env for the HF Spaces demo)
# ---------------------------------------------------------------------------

TASK_NAME = os.getenv("TASK_NAME", "easy").lower()

def _get_difficulty(task: str) -> TaskDifficulty:
    if task in ["easy", "easy_task"]:
        return TaskDifficulty.EASY
    elif task in ["medium", "medium_task"]:
        return TaskDifficulty.MEDIUM
    return TaskDifficulty.HARD


# Initialise once; callers use /reset to start a fresh episode
_env = ContentRecEnv(task_difficulty=_get_difficulty(TASK_NAME), seed=42)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class StepRequest(BaseModel):
    recommended_items: List[int]   # exactly 5 unique item IDs in [0, 499]
    task: Optional[str] = None     # optionally switch task on the fly


class ResetRequest(BaseModel):
    task: Optional[str] = None     # "easy" | "medium" | "hard"
    seed: Optional[int] = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_asdict(obj) -> dict:
    """Convert dataclass to dict, truncating large embedding lists."""
    d = asdict(obj)
    # Truncate 500-item catalog in observation to save bandwidth
    if "available_items" in d:
        for item in d["available_items"]:
            if "embedding" in item:
                item["embedding"] = item["embedding"][:4]  # first 4 dims only
    return d


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Liveness check")
def health():
    """Returns HTTP 200 with status OK. Used by HF pre-submission checker."""
    return {"status": "ok", "environment": "content-recommendation", "version": "1.0.0"}


@app.post("/reset", summary="Reset environment for a new episode")
async def reset(req: ResetRequest = ResetRequest()):
    """
    Resets the environment and returns the initial observation.
    Optionally switch to a different task difficulty via `task` field.
    """
    global _env
    difficulty = _get_difficulty(req.task or TASK_NAME)
    seed = req.seed if req.seed is not None else 42

    _env = ContentRecEnv(task_difficulty=difficulty, seed=seed)
    result = await _env.reset()

    return {
        "observation": _safe_asdict(result.observation),
        "reward": result.reward,
        "done": result.done,
        "task": difficulty.value,
    }


@app.post("/step", summary="Execute one environment step")
async def step(req: StepRequest):
    """
    Takes an action (5 recommended item IDs) and advances the environment.
    Returns the next observation, reward, and done flag.
    """
    try:
        action = RecommendationAction(recommended_items=req.recommended_items)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        result = await _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "observation": _safe_asdict(result.observation),
        "reward": round(result.reward, 4),
        "done": result.done,
        "info": result.info or {},
    }


@app.get("/state", summary="Get current user state")
def state():
    """Returns the current user state without advancing the episode."""
    try:
        user_state = _env.state()
        return asdict(user_state)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/episode_summary", summary="Get current episode summary")
def episode_summary():
    """Returns cumulative metrics for the ongoing episode."""
    return _env.get_episode_summary()


@app.get("/", summary="API root")
def root():
    return {
        "name": "Content Recommendation OpenEnv",
        "endpoints": ["/health", "/reset", "/step", "/state", "/episode_summary"],
        "docs": "/docs",
    }
