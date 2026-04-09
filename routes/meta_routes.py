"""
routes/meta_routes.py — System / discovery endpoints.

Endpoints:
  GET /           — API root
  GET /health     — Liveness check (required by HF Spaces + openenv validate)
  GET /metadata   — Environment metadata (required by openenv validate)
  GET /schema     — Action / observation / state JSON schemas
  GET /tasks      — List all tasks with grader status
"""

from fastapi import APIRouter

from models import ActionModel, HealthModel, ObservationModel, UserStateModel

router = APIRouter(tags=["System & Discovery"])


@router.get("/", summary="API root")
def root():
    """Returns a directory of all available endpoints."""
    return {
        "name": "Content Recommendation OpenEnv",
        "version": "1.0.0",
        "openenv_spec": True,
        "tasks_with_graders": 3,
        "endpoints": {
            "health":     "GET  /health",
            "metadata":   "GET  /metadata",
            "schema":     "GET  /schema",
            "tasks":      "GET  /tasks",
            "graders":    "GET  /graders",
            "reset":      "POST /reset",
            "step":       "POST /step",
            "state":      "GET  /state",
            "catalog":    "GET  /catalog",
            "grade":      "POST /grade",
            "grade_task": "GET  /grade/{task_name}",
            "grade_all":  "POST /grade/all",
            "grader":     "GET  /grader",
            "docs":       "GET  /docs",
        },
    }


@router.get("/health", response_model=HealthModel, summary="Liveness check")
def health():
    """Returns HTTP 200 with status=healthy. Required by HF Spaces and openenv validate."""
    return HealthModel(
        status="healthy",
        environment="content-recommendation",
        version="1.0.0",
    )


@router.get("/metadata", summary="Environment metadata")
def metadata():
    """
    Returns name, description, task list and grader references.
    Required by openenv validate runtime check.
    """
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


@router.get("/schema", summary="Action, observation, and state JSON schemas")
def schema():
    """Returns JSON schemas for action, observation, and state. Required by openenv validate."""
    return {
        "action":      ActionModel.model_json_schema(),
        "observation": ObservationModel.model_json_schema(),
        "state":       UserStateModel.model_json_schema(),
    }


@router.get("/tasks", summary="List all tasks with grader status")
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
                "description": "Pure CTR optimisation — reward = clicks / 5",
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
                "description": "CTR + content diversity penalty — requires balanced exploration",
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
                "description": "Long-term user lifetime value with churn prevention",
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
