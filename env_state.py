"""
env_state.py — Shared mutable state and helper utilities.

Centralises all global state so route modules stay stateless and testable.
Imported by routes/env_routes.py and routes/grader_routes.py.
"""

import os
import random
import uuid
from typing import Dict, List, Optional, Tuple

from content_rec_env import ContentCatalog, ContentRecEnv, TaskDifficulty
from models import (
    ItemSummaryModel,
    ObservationModel,
    RewardModel,
    UserStateModel,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TASK_NAME: str = os.getenv("TASK_NAME", "easy").lower()

# ---------------------------------------------------------------------------
# Session tracking — mirrors advaya's session-based grader architecture
# ---------------------------------------------------------------------------

SESSIONS: Dict[str, ContentRecEnv] = {}           # session_id → env
SESSION_TASKS: Dict[str, str] = {}                # session_id → task name
SESSION_ACTIONS: Dict[str, List[List[int]]] = {}  # session_id → actions taken

# ---------------------------------------------------------------------------
# Global env (backward compatibility for stateless clients)
# ---------------------------------------------------------------------------

_env: Optional[ContentRecEnv] = None


def get_env() -> Optional[ContentRecEnv]:
    return _env


def set_env(new_env: ContentRecEnv) -> None:
    global _env
    _env = new_env


# ---------------------------------------------------------------------------
# Difficulty resolver
# ---------------------------------------------------------------------------

def get_difficulty(task: str) -> TaskDifficulty:
    """Map task name string → TaskDifficulty enum."""
    t = (task or "easy").lower()
    if t in ("easy", "easy_task"):
        return TaskDifficulty.EASY
    if t in ("medium", "medium_task"):
        return TaskDifficulty.MEDIUM
    return TaskDifficulty.HARD


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def create_session(task: str, seed: int) -> Tuple[str, ContentRecEnv]:
    """
    Create a new episode session.
    Returns (session_id, fresh ContentRecEnv).
    Also updates the global _env for backward-compat clients.
    """
    difficulty = get_difficulty(task)
    new_env = ContentRecEnv(task_difficulty=difficulty, seed=seed)
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = new_env
    SESSION_TASKS[session_id] = task.lower()
    SESSION_ACTIONS[session_id] = []
    set_env(new_env)
    return session_id, new_env


def record_action(session_id: Optional[str], action: List[int]) -> None:
    """Append an action to the session's action log (for grader replay)."""
    if session_id and session_id in SESSION_ACTIONS:
        SESSION_ACTIONS[session_id].append(action)


def get_session_actions(session_id: Optional[str]) -> Optional[List[List[int]]]:
    """Return recorded actions for a session, or None if session not found."""
    if session_id and session_id in SESSION_ACTIONS:
        acts = SESSION_ACTIONS[session_id]
        return acts if acts else None
    return None


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def obs_to_model(obs_dc) -> ObservationModel:
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
        segment=getattr(us, "segment", "casual"),
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


def info_to_reward_model(info: Optional[Dict]) -> RewardModel:
    """Convert step info dict → typed RewardModel."""
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
        session_value=info.get("session_value"),
    )


# ---------------------------------------------------------------------------
# Auto-action generator (for self-play grading)
# ---------------------------------------------------------------------------

def auto_actions(seed: int = 42, n_steps: int = 10) -> List[List[int]]:
    """
    Generate a set of heuristic-quality actions for self-play grading.
    Picks popular + fresh items — better than random, useful as a baseline.
    """
    rng = random.Random(seed)
    catalog = ContentCatalog(seed=seed)
    items = catalog.get_all()
    # Pre-rank by popularity + freshness
    pool = sorted(
        items,
        key=lambda x: x.popularity + (1.0 - x.freshness / 90.0) * 0.5,
        reverse=True,
    )
    top = pool[:30]  # draw from top 30 only
    actions = []
    for _ in range(n_steps):
        sample = rng.sample(top, 5)
        actions.append([it.item_id for it in sample])
    return actions


# ---------------------------------------------------------------------------
# Initialise global env at import time
# ---------------------------------------------------------------------------

set_env(ContentRecEnv(task_difficulty=get_difficulty(TASK_NAME), seed=42))
