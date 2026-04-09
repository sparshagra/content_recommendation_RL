"""
routes/grader_routes.py — Grading and evaluation endpoints.

Endpoints:
  GET  /grader          — Session-based grader (called by inference.py after each episode)
  GET  /graders         — List all available graders
  POST /grade           — Grade a task with provided or auto-generated actions
  GET  /grade/{task}    — Grade a task via GET (no body needed)
  POST /grade/all       — Grade all three tasks at once
"""

from typing import Optional

from fastapi import APIRouter, HTTPException

from env_state import (
    SESSION_TASKS,
    TASK_NAME,
    auto_actions,
    get_session_actions,
)
from graders import easy_task_grader, grade_all, hard_task_grader, medium_task_grader
from models import GradeAllRequest, GradeRequest, GradeResult

router = APIRouter(tags=["Grading"])

# Map task name variants → (grader_fn, module_ref)
GRADER_MAP = {
    "easy":        (easy_task_grader,   "graders.easy_task_grader"),
    "easy_task":   (easy_task_grader,   "graders.easy_task_grader"),
    "medium":      (medium_task_grader, "graders.medium_task_grader"),
    "medium_task": (medium_task_grader, "graders.medium_task_grader"),
    "hard":        (hard_task_grader,   "graders.hard_task_grader"),
    "hard_task":   (hard_task_grader,   "graders.hard_task_grader"),
}


@router.get("/grader", summary="Grade a completed session (OpenEnv evaluator endpoint)")
def grader_session(session_id: Optional[str] = None, task: Optional[str] = None):
    """
    Session-based grader — called by inference.py after each episode.

    The OpenEnv evaluator calls this to verify graders exist and are functional.
    If session_id is provided and matches a recorded session, the agent's actual
    actions are graded. Otherwise a deterministic self-play episode is run.
    """
    # Resolve task name from session or parameter
    task_name = task or TASK_NAME
    if session_id and session_id in SESSION_TASKS:
        task_name = SESSION_TASKS[session_id]

    entry = GRADER_MAP.get((task_name or "easy").lower())
    if not entry:
        entry = GRADER_MAP["easy"]
    grader_fn, _ = entry

    # Use recorded session actions if available, else self-play
    actions = get_session_actions(session_id)
    if actions is None:
        actions = auto_actions(seed=42)

    result = grader_fn(actions=actions)
    result["session_id"] = session_id
    result["grader_type"] = "deterministic"
    return result


@router.get("/graders", summary="List all task graders")
def graders():
    """
    Lists all available grader functions and their task associations.
    Required by OpenEnv Phase 2 grader discovery.
    """
    return {
        "graders": [
            {
                "task":              "easy_task",
                "grader":            "graders.easy_task_grader",
                "grader_type":       "deterministic",
                "score_range":       [0.0, 1.0],
                "success_threshold": 0.25,
                "metric":            "average_ctr",
                "callable":          True,
                "endpoint":          "/grade/easy_task",
            },
            {
                "task":              "medium_task",
                "grader":            "graders.medium_task_grader",
                "grader_type":       "deterministic",
                "score_range":       [0.0, 1.0],
                "success_threshold": 0.15,
                "metric":            "engagement_minus_diversity_penalty",
                "callable":          True,
                "endpoint":          "/grade/medium_task",
            },
            {
                "task":              "hard_task",
                "grader":            "graders.hard_task_grader",
                "grader_type":       "deterministic",
                "score_range":       [0.0, 1.0],
                "success_threshold": 0.20,
                "metric":            "lifetime_value_with_churn",
                "callable":          True,
                "endpoint":          "/grade/hard_task",
            },
        ],
        "total_graders": 3,
        "graders_callable": True,
    }


@router.post("/grade", response_model=GradeResult, summary="Grade a task")
def grade(req: GradeRequest):
    """
    Grade a sequence of agent actions for a specific task.

    - Provide `actions` (list of 5-item lists) to grade your agent's actual behaviour.
    - Omit `actions` to run a deterministic self-play episode as a baseline.

    Returns a deterministic score in [0.0, 1.0]. Required by OpenEnv Phase 2.
    """
    entry = GRADER_MAP.get(req.task.lower())
    if not entry:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{req.task}'. Must be one of: easy_task, medium_task, hard_task",
        )
    grader_fn, grader_ref = entry
    seed = req.seed if req.seed is not None else 42
    actions = req.actions if req.actions else auto_actions(seed=seed)
    result = grader_fn(actions, seed=seed)
    result["grader"] = grader_ref
    result["grader_type"] = "deterministic"
    result.setdefault("passed", len(result.get("errors", [])) == 0)
    return GradeResult(**result)


@router.get(
    "/grade/{task_name}",
    response_model=GradeResult,
    summary="Grade a task via GET (no body required)",
)
def grade_get(task_name: str, seed: int = 42):
    """
    Grade a task with a single GET request — runs a deterministic self-play episode.
    Useful for quick baseline checks without constructing a request body.

    task_name: easy_task | medium_task | hard_task (or easy | medium | hard)
    """
    return grade(GradeRequest(task=task_name, actions=None, seed=seed))


@router.post("/grade/all", summary="Grade all three tasks at once")
def grade_all_endpoint(req: Optional[GradeAllRequest] = None):
    """
    Grade agent actions for all tasks in one request.

    Returns per-task scores and an aggregate overall score.
    Omit or leave `actions` null to auto-grade all tasks using self-play.
    """
    seed = req.seed if req and req.seed is not None else 42
    if req and req.actions:
        return grade_all(req.actions, seed=seed)
    shared = auto_actions(seed=seed)
    return grade_all(
        {"easy_task": shared, "medium_task": shared, "hard_task": shared},
        seed=seed,
    )
