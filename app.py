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
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from content_rec_env import ContentRecEnv, RecommendationAction, TaskDifficulty
from graders import easy_task_grader, medium_task_grader, hard_task_grader, grade_all

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
    session_id: str = Field("", description="Session ID — pass to /grader to score this episode")


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
    """POST /step request body — extends ActionModel with optional session_id."""
    session_id: Optional[str] = Field(None, description="Session ID from /reset (optional)")


class ResetRequest(BaseModel):
    task: Optional[str] = Field(None, description="easy | medium | hard")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")


class GradeRequest(BaseModel):
    """POST /grade request body. actions is optional — if omitted, a deterministic self-play episode is run."""
    task: str = Field(..., description="Task to grade: easy_task | medium_task | hard_task")
    actions: Optional[List[List[int]]] = Field(
        None, description="List of actions (each a list of 5 item IDs). If omitted, a deterministic episode is auto-run."
    )
    seed: Optional[int] = Field(42, description="Random seed")


class GradeAllRequest(BaseModel):
    """POST /grade/all request body. actions is optional."""
    actions: Optional[Dict[str, List[List[int]]]] = Field(
        None, description="Dict mapping task name to list of actions. If omitted, all tasks are auto-graded."
    )
    seed: Optional[int] = Field(42, description="Random seed")


class GradeResult(BaseModel):
    """Grade result for a single task."""
    task: str
    score: float = Field(..., ge=0.0, le=1.0)
    rewards: List[float]
    steps: int
    success: bool
    passed: bool = True
    metric: Optional[str] = None
    success_threshold: Optional[float] = None
    errors: List[str] = []
    grader: Optional[str] = None
    grader_type: str = "deterministic"


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------

TASK_NAME = os.getenv("TASK_NAME", "easy").lower()

# ---------------------------------------------------------------------------
# Session tracking — mirrors advaya's session-based grader architecture
# ---------------------------------------------------------------------------

SESSIONS: Dict[str, ContentRecEnv] = {}      # session_id → env
SESSION_TASKS: Dict[str, str] = {}           # session_id → task name
SESSION_ACTIONS: Dict[str, List[List[int]]] = {}  # session_id → list of actions taken


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
    Reset environment for a new episode. Returns the initial typed Observation
    and a session_id for use with /step and /grader.
    """
    global _env
    difficulty = _get_difficulty(req.task or TASK_NAME)
    seed = req.seed if req.seed is not None else 42
    new_env = ContentRecEnv(task_difficulty=difficulty, seed=seed)
    result = await new_env.reset()

    # Create a session so /grader can score the completed episode
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = new_env
    SESSION_TASKS[session_id] = (req.task or TASK_NAME).lower()
    SESSION_ACTIONS[session_id] = []

    # Also update the global env for backward compat
    _env = new_env

    return ResetResponseModel(
        observation=_obs_to_model(result.observation),
        reward=result.reward,
        done=result.done,
        task=difficulty.value,
        session_id=session_id,
    )


@app.post("/step", response_model=StepResponseModel, summary="Execute one step")
async def step(req: StepRequest):
    """
    Submit a recommendation action (5 item IDs). Returns next Observation,
    Reward [0.0-1.0], done flag, and typed reward breakdown.
    Uses session_id if provided, otherwise falls back to global env.
    """
    # Resolve which env to use
    env_to_use = _env
    if req.session_id and req.session_id in SESSIONS:
        env_to_use = SESSIONS[req.session_id]
        # Record the action for grading later
        SESSION_ACTIONS[req.session_id].append(req.recommended_items)

    try:
        action = RecommendationAction(recommended_items=req.recommended_items)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    try:
        result = await env_to_use.step(action)
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
        "tasks": [
            {
                "name": "easy_task",
                "difficulty": "easy",
                "has_grader": True,
                "grader": "graders.easy_task_grader",
                "grader_type": "deterministic",
                "grader_endpoint": "/grade/easy_task",
                "success_threshold": 0.25,
                "score_range": [0.0, 1.0],
            },
            {
                "name": "medium_task",
                "difficulty": "medium",
                "has_grader": True,
                "grader": "graders.medium_task_grader",
                "grader_type": "deterministic",
                "grader_endpoint": "/grade/medium_task",
                "success_threshold": 0.15,
                "score_range": [0.0, 1.0],
            },
            {
                "name": "hard_task",
                "difficulty": "hard",
                "has_grader": True,
                "grader": "graders.hard_task_grader",
                "grader_type": "deterministic",
                "grader_endpoint": "/grade/hard_task",
                "success_threshold": 0.20,
                "score_range": [0.0, 1.0],
            },
        ],
        "tasks_with_graders": 3,
        "graders_endpoint": "/graders",
    }


@app.get("/schema", summary="Action, observation, and state schemas")
def schema():
    """Returns JSON schemas for action, observation, and state. Required by openenv validate."""
    return {
        "action": ActionModel.model_json_schema(),
        "observation": ObservationModel.model_json_schema(),
        "state": UserStateModel.model_json_schema(),
    }


def _auto_actions(seed: int = 42) -> List[List[int]]:
    """Generate a default set of 10 heuristic-quality actions for self-play grading."""
    import random as _rnd
    _rnd.seed(seed)
    from content_rec_env import ContentCatalog
    catalog = ContentCatalog(seed=seed)
    items = catalog.get_all()
    actions = []
    for _ in range(10):
        # pick 5 unique items with some score bias toward popular + fresh
        pool = sorted(items, key=lambda x: x.popularity + (1 - x.freshness / 90) * 0.5, reverse=True)
        chosen = pool[:20]
        _rnd.shuffle(chosen)
        ids = [it.item_id for it in chosen[:5]]
        actions.append(ids)
    return actions


@app.post("/grade", response_model=GradeResult, summary="Grade a task")
def grade(req: GradeRequest):
    """
    Grade a sequence of agent actions for a specific task.
    Returns deterministic score in [0.0, 1.0].
    If no actions are provided, a deterministic self-play episode is auto-run.
    Required by OpenEnv Phase 2.
    """
    grader_map = {
        "easy_task": (easy_task_grader, "graders.easy_task_grader"),
        "easy": (easy_task_grader, "graders.easy_task_grader"),
        "medium_task": (medium_task_grader, "graders.medium_task_grader"),
        "medium": (medium_task_grader, "graders.medium_task_grader"),
        "hard_task": (hard_task_grader, "graders.hard_task_grader"),
        "hard": (hard_task_grader, "graders.hard_task_grader"),
    }
    task = req.task.lower()
    entry = grader_map.get(task)
    if not entry:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{req.task}'. Must be one of: easy_task, medium_task, hard_task",
        )
    grader_fn, grader_ref = entry
    seed = req.seed if req.seed is not None else 42
    # Use provided actions or auto-generate
    actions = req.actions if req.actions else _auto_actions(seed=seed)
    result = grader_fn(actions, seed=seed)
    result["grader"] = grader_ref
    result["grader_type"] = "deterministic"
    result.setdefault("passed", result.get("passed", len(result.get("errors", [])) == 0))
    return GradeResult(**result)


@app.get("/grade/{task_name}", response_model=GradeResult, summary="Grade a task (no body required)")
def grade_get(task_name: str, seed: int = 42):
    """
    Grade a task via GET — runs a deterministic self-play episode automatically.
    task_name: easy_task | medium_task | hard_task (or easy | medium | hard)
    """
    from pydantic import ValidationError
    req = GradeRequest(task=task_name, actions=None, seed=seed)
    return grade(req)


@app.get("/grader", summary="Grade a completed session (session-based grader endpoint)")
def grader_session(
    session_id: str = None,
    task: str = None,
):
    """
    Session-based grader — called by inference.py after each episode.
    Matches the advaya/OpenEnv evaluator's expected grader pattern.
    Grades the actual actions taken in the session (or runs a deterministic
    episode if no session is found).
    """
    # Resolve task name
    task_name = task or TASK_NAME
    if session_id and session_id in SESSION_TASKS:
        task_name = SESSION_TASKS[session_id]

    grader_map = {
        "easy": easy_task_grader,       "easy_task": easy_task_grader,
        "medium": medium_task_grader,   "medium_task": medium_task_grader,
        "hard": hard_task_grader,       "hard_task": hard_task_grader,
    }
    grader_fn = grader_map.get((task_name or "easy").lower(), easy_task_grader)

    # Grade the actual session actions if available
    actions = None
    if session_id and session_id in SESSION_ACTIONS and SESSION_ACTIONS[session_id]:
        actions = SESSION_ACTIONS[session_id]

    result = grader_fn(actions=actions)
    result["session_id"] = session_id
    result["grader_type"] = "deterministic"
    return result


@app.get("/graders", summary="List all task graders")
def graders():
    """
    Lists all available grader functions and their task associations.
    Required by OpenEnv Phase 2 grader discovery.
    """
    return {
        "graders": [
            {
                "task": "easy_task",
                "grader": "graders.easy_task_grader",
                "grader_type": "deterministic",
                "score_range": [0.0, 1.0],
                "success_threshold": 0.25,
                "metric": "average_ctr",
                "callable": True,
                "endpoint": "/grade/easy_task",
            },
            {
                "task": "medium_task",
                "grader": "graders.medium_task_grader",
                "grader_type": "deterministic",
                "score_range": [0.0, 1.0],
                "success_threshold": 0.15,
                "metric": "engagement_minus_diversity_penalty",
                "callable": True,
                "endpoint": "/grade/medium_task",
            },
            {
                "task": "hard_task",
                "grader": "graders.hard_task_grader",
                "grader_type": "deterministic",
                "score_range": [0.0, 1.0],
                "success_threshold": 0.20,
                "metric": "lifetime_value_with_churn",
                "callable": True,
                "endpoint": "/grade/hard_task",
            },
        ],
        "total_graders": 3,
        "graders_callable": True,
    }


@app.post("/grade/all", summary="Grade all tasks")
def grade_all_endpoint(req: Optional[GradeAllRequest] = None):
    """
    Grade agent actions for all tasks at once.
    Returns per-task scores and an overall score.
    If no actions provided, all tasks are auto-graded using self-play.
    """
    seed = (req.seed if req and req.seed is not None else 42)
    if req and req.actions:
        return grade_all(req.actions, seed=seed)
    # Auto-grade all tasks
    auto_actions = _auto_actions(seed=seed)
    return grade_all(
        {
            "easy_task": auto_actions,
            "medium_task": auto_actions,
            "hard_task": auto_actions,
        },
        seed=seed,
    )


@app.get("/tasks", summary="List all tasks with graders")
def tasks():
    """
    Lists all available tasks and their grader status.
    Required by OpenEnv Phase 2 evaluation.
    """
    return {
        "tasks": [
            {
                "name": "easy_task",
                "difficulty": "easy",
                "description": "Pure CTR optimization",
                "grader": "graders.easy_task_grader",
                "grader_type": "deterministic",
                "grader_endpoint": "/grade/easy_task",
                "success_threshold": 0.25,
                "score_range": [0.0, 1.0],
                "has_grader": True,
            },
            {
                "name": "medium_task",
                "difficulty": "medium",
                "description": "Engagement with diversity penalty",
                "grader": "graders.medium_task_grader",
                "grader_type": "deterministic",
                "grader_endpoint": "/grade/medium_task",
                "success_threshold": 0.15,
                "score_range": [0.0, 1.0],
                "has_grader": True,
            },
            {
                "name": "hard_task",
                "difficulty": "hard",
                "description": "Long-term lifetime value with churn prevention",
                "grader": "graders.hard_task_grader",
                "grader_type": "deterministic",
                "grader_endpoint": "/grade/hard_task",
                "success_threshold": 0.20,
                "score_range": [0.0, 1.0],
                "has_grader": True,
            },
        ],
        "total_tasks": 3,
        "tasks_with_graders": 3,
    }


@app.get("/", summary="API root")
def root():
    return {
        "name": "Content Recommendation OpenEnv",
        "version": "1.0.0",
        "openenv_spec": True,
        "tasks_with_graders": 3,
        "endpoints": {
            "health": "GET /health",
            "metadata": "GET /metadata",
            "schema": "GET /schema",
            "tasks": "GET /tasks",
            "graders": "GET /graders",
            "reset": "POST /reset",
            "step": "POST /step",
            "state": "GET /state",
            "catalog": "GET /catalog",
            "grade": "POST /grade",
            "grade_task": "GET /grade/{task_name}",
            "grade_all": "POST /grade/all",
            "docs": "GET /docs",
        },
    }
