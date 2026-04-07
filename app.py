"""
Content Recommendation Environment — FastAPI Server
=====================================================
OpenEnv-compliant HTTP server for Hugging Face Spaces.

Endpoints:
  GET  /health              — liveness check (HTTP 200)
  POST /reset               — reset environment, returns typed Observation
  POST /step                — execute action, returns typed StepResult
  GET  /state               — current user state
  GET  /catalog             — full item catalog (500 items)
  GET  /episode_summary     — cumulative episode metrics
  GET  /docs                — Swagger/OpenAPI documentation

All response models are typed Pydantic models per OpenEnv specification.
"""

import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from content_rec_env import ContentRecEnv, RecommendationAction, TaskDifficulty

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Content Recommendation OpenEnv",
    description=(
        "Real-world RL environment for content recommendation. "
        "An agent recommends 5 items per step and receives reward signals "
        "based on click-through rate, diversity, and long-term user retention. "
        "Implements the full OpenEnv spec: typed models, step/reset/state API."
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
# Typed Pydantic Models — OpenEnv Specification Compliance
# ---------------------------------------------------------------------------

class ActionModel(BaseModel):
    """Agent action: recommend exactly 5 unique item IDs from the catalog."""
    recommended_items: List[int] = Field(
        ...,
        description="Exactly 5 unique item IDs in range [0, 499]",
        min_length=5,
        max_length=5,
    )


class UserStateModel(BaseModel):
    """Typed observation of the current user state."""
    user_id: int = Field(..., description="Unique user ID (0-999)")
    session_id: int = Field(..., description="Unique session identifier per episode")
    interaction_history: List[int] = Field(
        ..., description="Last 5 item IDs clicked (empty list if none)"
    )
    genre_preferences: Dict[str, float] = Field(
        ..., description="Genre → preference score mapping (0.0-1.0)"
    )
    satisfaction: float = Field(..., ge=0.0, le=1.0, description="Session satisfaction")
    fatigue: float = Field(..., ge=0.0, le=1.0, description="Content fatigue level")
    churn_risk: float = Field(..., ge=0.0, le=1.0, description="Churn probability")
    session_step: int = Field(..., description="Current step in session (0-10)")


class ItemSummaryModel(BaseModel):
    """Compact content item (embedding omitted to reduce payload size)."""
    item_id: int
    title: str
    genre: str = Field(..., description="One of: pop, tech, sports, entertainment, news")
    popularity: float = Field(..., ge=0.0, le=1.0)
    freshness: int = Field(..., description="Days old (0-90)")
    duration: int = Field(..., description="Content length in seconds")


class ObservationModel(BaseModel):
    """Typed observation returned by reset() and step(). OpenEnv spec compliant."""
    user_state: UserStateModel
    available_items: List[ItemSummaryModel] = Field(
        ..., description="All 500 catalog items (embedding excluded)"
    )
    last_action: Optional[List[int]] = Field(
        None, description="Item IDs recommended in previous step (null on first step)"
    )
    last_clicks: Optional[List[int]] = Field(
        None, description="Items clicked in previous step (null on first step)"
    )
    last_reward: float = Field(0.0, description="Reward from previous step")


class RewardModel(BaseModel):
    """Typed reward info returned by step(). Breakdown per task type."""
    ctr: Optional[float] = Field(None, description="Click-through rate [0, 1]")
    clicks: Optional[int] = Field(None, description="Number of clicked items")
    task: Optional[str] = Field(None, description="Task name")
    genre_diversity: Optional[float] = Field(None, description="Genre diversity [0, 1]")
    diversity_penalty: Optional[float] = Field(None, description="Diversity penalty (medium/hard)")
    satisfaction_bonus: Optional[float] = Field(None, description="Satisfaction bonus (hard)")
    churn_penalty: Optional[float] = Field(None, description="Churn risk penalty (hard)")
    diversity_bonus: Optional[float] = Field(None, description="Diversity bonus (hard)")
    churn_risk: Optional[float] = Field(None, description="Current churn risk")
    satisfaction: Optional[float] = Field(None, description="Current satisfaction")


class ResetResponseModel(BaseModel):
    """Response from POST /reset."""
    observation: ObservationModel
    reward: float = Field(0.0, ge=0.0, le=1.0)
    done: bool = False
    task: str = Field(..., description="Active task: easy | medium | hard")


class StepResponseModel(BaseModel):
    """Response from POST /step."""
    observation: ObservationModel
    reward: float = Field(..., ge=0.0, le=1.0, description="Reward in [0.0, 1.0]")
    done: bool
    info: RewardModel = Field(..., description="Detailed reward breakdown")


class HealthModel(BaseModel):
    status: str
    environment: str
    version: str


class EpisodeSummaryModel(BaseModel):
    task: str
    steps: int
    total_reward: float
    avg_reward: float
    rewards: List[float]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class StepRequest(ActionModel):
    """POST /step request body — extends ActionModel."""
    pass


class ResetRequest(BaseModel):
    task: Optional[str] = Field(None, description="easy | medium | hard")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

TASK_NAME = os.getenv("TASK_NAME", "easy").lower()


def _get_difficulty(task: str) -> TaskDifficulty:
    t = (task or "easy").lower()
    if t in ("easy", "easy_task"):
        return TaskDifficulty.EASY
    if t in ("medium", "medium_task"):
        return TaskDifficulty.MEDIUM
    return TaskDifficulty.HARD


_env = ContentRecEnv(task_difficulty=_get_difficulty(TASK_NAME), seed=42)


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _obs_to_model(obs_dc) -> ObservationModel:
    """Convert ContentRecObservation dataclass → typed ObservationModel."""
    us = obs_dc.user_state
    user_state = UserStateModel(
        user_id=us.user_id,
        session_id=us.session_id,
        interaction_history=us.interaction_history,
        genre_preferences=us.genre_preferences,
        satisfaction=round(us.satisfaction, 4),
        fatigue=round(us.fatigue, 4),
        churn_risk=round(us.churn_risk, 4),
        session_step=us.session_step,
    )
    items = [
        ItemSummaryModel(
            item_id=it.item_id,
            title=it.title,
            genre=it.genre,
            popularity=round(it.popularity, 4),
            freshness=it.freshness,
            duration=it.duration,
        )
        for it in obs_dc.available_items
    ]
    return ObservationModel(
        user_state=user_state,
        available_items=items,
        last_action=obs_dc.last_action,
        last_clicks=obs_dc.last_clicks,
        last_reward=round(obs_dc.last_reward, 4),
    )


def _info_to_reward_model(info: Optional[Dict]) -> RewardModel:
    if not info:
        return RewardModel()
    return RewardModel(
        ctr=info.get("ctr"),
        clicks=info.get("clicks"),
        task=info.get("task"),
        genre_diversity=info.get("genre_diversity"),
        diversity_penalty=info.get("diversity_penalty"),
        satisfaction_bonus=info.get("satisfaction_bonus"),
        churn_penalty=info.get("churn_penalty"),
        diversity_bonus=info.get("diversity_bonus"),
        churn_risk=info.get("churn_risk"),
        satisfaction=info.get("satisfaction"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthModel, summary="Liveness check")
def health():
    """Returns HTTP 200. Required by HF Spaces and openenv validate."""
    return HealthModel(
        status="healthy",
        environment="content-recommendation",
        version="1.0.0",
    )


@app.post("/reset", response_model=ResetResponseModel, summary="Reset environment")
async def reset(req: ResetRequest = ResetRequest()):
    """
    Reset environment for a new episode. Returns the initial typed Observation.
    Optionally specify `task` (easy/medium/hard) and `seed`.
    """
    global _env
    difficulty = _get_difficulty(req.task or TASK_NAME)
    seed = req.seed if req.seed is not None else 42
    _env = ContentRecEnv(task_difficulty=difficulty, seed=seed)
    result = await _env.reset()

    return ResetResponseModel(
        observation=_obs_to_model(result.observation),
        reward=result.reward,
        done=result.done,
        task=difficulty.value,
    )


@app.post("/step", response_model=StepResponseModel, summary="Execute one step")
async def step(req: StepRequest):
    """
    Submit a recommendation action (5 item IDs). Returns next Observation,
    Reward [0.0-1.0], done flag, and typed reward breakdown.
    """
    try:
        action = RecommendationAction(recommended_items=req.recommended_items)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    try:
        result = await _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponseModel(
        observation=_obs_to_model(result.observation),
        reward=round(max(0.0, min(1.0, result.reward)), 4),
        done=result.done,
        info=_info_to_reward_model(result.info),
    )


@app.get("/state", response_model=UserStateModel, summary="Get current user state")
def state():
    """Returns the current UserState without advancing the episode."""
    try:
        us = _env.state()
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return UserStateModel(
        user_id=us.user_id,
        session_id=us.session_id,
        interaction_history=us.interaction_history,
        genre_preferences=us.genre_preferences,
        satisfaction=round(us.satisfaction, 4),
        fatigue=round(us.fatigue, 4),
        churn_risk=round(us.churn_risk, 4),
        session_step=us.session_step,
    )


@app.get(
    "/catalog",
    response_model=List[ItemSummaryModel],
    summary="List all catalog items",
)
def catalog():
    """Returns all 500 items in the content catalog (without embeddings)."""
    from content_rec_env import ContentCatalog
    cat = ContentCatalog(seed=42)
    return [
        ItemSummaryModel(
            item_id=it.item_id,
            title=it.title,
            genre=it.genre,
            popularity=round(it.popularity, 4),
            freshness=it.freshness,
            duration=it.duration,
        )
        for it in cat.get_all()
    ]


@app.get(
    "/episode_summary",
    response_model=EpisodeSummaryModel,
    summary="Current episode metrics",
)
def episode_summary():
    """Returns cumulative reward and step info for the current episode."""
    s = _env.get_episode_summary()
    return EpisodeSummaryModel(
        task=s["task"],
        steps=s["steps"],
        total_reward=round(s["total_reward"], 4),
        avg_reward=round(s["avg_reward"], 4),
        rewards=[round(r, 4) for r in s["rewards"]],
    )


@app.get("/metadata", summary="Environment metadata")
def metadata():
    """Returns name and description. Required by openenv validate runtime check."""
    return {
        "name": "content-recommendation",
        "description": (
            "A real-world RL environment for content recommendation. "
            "An agent recommends 5 items per step to simulated users and "
            "receives reward signals based on click-through rate, diversity, "
            "and long-term user retention."
        ),
        "version": "1.0.0",
        "author": "dikshi2025",
        "tasks": ["easy", "medium", "hard"],
    }


@app.get("/schema", summary="Action, observation, and state schemas")
def schema():
    """Returns JSON schemas for action, observation, and state. Required by openenv validate."""
    return {
        "action": ActionModel.model_json_schema(),
        "observation": ObservationModel.model_json_schema(),
        "state": UserStateModel.model_json_schema(),
    }


@app.get("/", summary="API root")
def root():
    return {
        "name": "Content Recommendation OpenEnv",
        "version": "1.0.0",
        "openenv_spec": True,
        "endpoints": {
            "health": "GET /health",
            "metadata": "GET /metadata",
            "schema": "GET /schema",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "catalog": "GET /catalog",
            "docs": "GET /docs",
        },
    }
