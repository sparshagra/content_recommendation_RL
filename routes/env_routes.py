"""
routes/env_routes.py — RL environment interaction endpoints.

Endpoints:
  POST /reset           — Start a new episode, returns initial observation + session_id
  POST /step            — Execute one action, returns observation + reward
  GET  /state           — Current user state (no side effects)
  GET  /catalog         — Full item catalog (500 items, no embeddings)
  GET  /episode_summary — Cumulative metrics for the current episode
"""

from typing import List

from fastapi import APIRouter, HTTPException

from content_rec_env import ContentCatalog, RecommendationAction
from env_state import (
    SESSIONS,
    TASK_NAME,
    create_session,
    get_env,
    info_to_reward_model,
    obs_to_model,
    record_action,
)
from models import (
    EpisodeSummaryModel,
    ItemSummaryModel,
    ResetRequest,
    ResetResponseModel,
    StepRequest,
    StepResponseModel,
    UserStateModel,
)

router = APIRouter(tags=["Environment"])


@router.post("/reset", response_model=ResetResponseModel, summary="Reset environment")
async def reset(req: ResetRequest = ResetRequest()):
    """
    Reset the environment for a new episode.

    Returns the initial typed Observation and a **session_id**.
    Pass session_id to subsequent /step calls and to /grader at the end
    to have the agent's actual actions graded.
    """
    task = req.task or TASK_NAME
    seed = req.seed if req.seed is not None else 42

    session_id, new_env = create_session(task, seed)
    result = await new_env.reset()

    return ResetResponseModel(
        observation=obs_to_model(result.observation),
        reward=result.reward,
        done=result.done,
        task=new_env.task_difficulty.value,
        session_id=session_id,
    )


@router.post("/step", response_model=StepResponseModel, summary="Execute one step")
async def step(req: StepRequest):
    """
    Submit a recommendation action — 5 unique item IDs.

    Returns the next observation, reward [0.0–1.0], done flag, and a full
    reward breakdown. Pass **session_id** from /reset so that actions are
    recorded for grader replay via /grader.
    """
    # Resolve session env (falls back to global env for stateless clients)
    env_to_use = get_env()
    if req.session_id and req.session_id in SESSIONS:
        env_to_use = SESSIONS[req.session_id]

    if env_to_use is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")

    # Record action for grader replay
    record_action(req.session_id, req.recommended_items)

    try:
        action = RecommendationAction(recommended_items=req.recommended_items)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    try:
        result = await env_to_use.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return StepResponseModel(
        observation=obs_to_model(result.observation),
        reward=round(max(0.0, min(1.0, result.reward)), 4),
        done=result.done,
        info=info_to_reward_model(result.info),
    )


@router.get("/state", response_model=UserStateModel, summary="Current user state")
def state():
    """Returns the current UserState without advancing the episode (zero side effects)."""
    env = get_env()
    if env is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    try:
        us = env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

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


@router.get("/catalog", response_model=List[ItemSummaryModel], summary="Full item catalog")
def catalog():
    """
    Returns all 500 items in the content catalog (embeddings excluded).
    Use this to inspect the action space before calling /reset.
    """
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


@router.get(
    "/episode_summary",
    response_model=EpisodeSummaryModel,
    summary="Current episode metrics",
)
def episode_summary():
    """Returns cumulative reward and step info for the current episode."""
    env = get_env()
    if env is None:
        raise HTTPException(status_code=400, detail="No active session. Call /reset first.")
    s = env.get_episode_summary()
    return EpisodeSummaryModel(
        task=s["task"],
        steps=s["steps"],
        total_reward=round(s["total_reward"], 4),
        avg_reward=round(s["avg_reward"], 4),
        rewards=[round(r, 4) for r in s["rewards"]],
    )
