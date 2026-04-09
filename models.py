"""
models.py — All Pydantic models for the Content Recommendation OpenEnv API.
Shared across route modules and the FastAPI app.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation / State models
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
    """Typed representation of the current simulated user's state."""
    user_id: int = Field(..., description="Unique user ID (0–999)")
    session_id: int = Field(..., description="Unique session identifier per episode")
    interaction_history: List[int] = Field(
        ..., description="Last 5 item IDs clicked (empty list if none)"
    )
    genre_preferences: Dict[str, float] = Field(
        ..., description="Genre → preference score mapping (0.0–1.0)"
    )
    satisfaction: float = Field(..., ge=0.0, le=1.0, description="Session satisfaction score")
    fatigue: float = Field(..., ge=0.0, le=1.0, description="Content fatigue level")
    churn_risk: float = Field(..., ge=0.0, le=1.0, description="Estimated churn probability")
    session_step: int = Field(..., description="Current step in session (0–max_steps)")


class ItemSummaryModel(BaseModel):
    """Compact content item summary (embedding omitted to reduce payload size)."""
    item_id: int = Field(..., description="Unique item ID (0–499)")
    title: str
    genre: str = Field(..., description="One of: pop, tech, sports, entertainment, news")
    popularity: float = Field(..., ge=0.0, le=1.0)
    freshness: int = Field(..., description="Days old (0–90)")
    duration: int = Field(..., description="Content length in seconds")


class ObservationModel(BaseModel):
    """Typed observation returned by /reset and /step. OpenEnv spec compliant."""
    user_state: UserStateModel
    available_items: List[ItemSummaryModel] = Field(
        ..., description="Top-50 candidate items pre-filtered for this user (embedding excluded)"
    )
    last_action: Optional[List[int]] = Field(
        None, description="Item IDs recommended in previous step (null on first step)"
    )
    last_clicks: Optional[List[int]] = Field(
        None, description="Items clicked in previous step (null on first step)"
    )
    last_reward: float = Field(0.0, description="Reward from previous step")


# ---------------------------------------------------------------------------
# Reward / Step models
# ---------------------------------------------------------------------------

class RewardModel(BaseModel):
    """Typed reward breakdown returned by /step. All fields task-dependent."""
    ctr: Optional[float] = Field(None, description="Click-through rate [0, 1]")
    clicks: Optional[int] = Field(None, description="Number of clicked items")
    task: Optional[str] = Field(None, description="Task name")
    genre_diversity: Optional[float] = Field(None, description="Genre diversity [0, 1]")
    diversity_penalty: Optional[float] = Field(None, description="Diversity penalty (medium/hard)")
    satisfaction_bonus: Optional[float] = Field(None, description="Satisfaction bonus (hard)")
    churn_penalty: Optional[float] = Field(None, description="Churn risk penalty (hard)")
    diversity_bonus: Optional[float] = Field(None, description="Diversity bonus (hard)")
    churn_risk: Optional[float] = Field(None, description="Current churn risk")
    satisfaction: Optional[float] = Field(None, description="Current satisfaction level")
    session_value: Optional[float] = Field(None, description="Business-interpretable session value")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

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
    reward: float = Field(..., ge=0.0, le=1.0, description="Step reward in [0.0, 1.0]")
    done: bool
    info: RewardModel = Field(..., description="Detailed reward breakdown by component")


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
    """POST /grade — grade a sequence of agent actions for a specific task."""
    task: str = Field(..., description="Task to grade: easy_task | medium_task | hard_task")
    actions: Optional[List[List[int]]] = Field(
        None,
        description="List of actions (each a list of 5 item IDs). "
                    "Omit to run a deterministic self-play episode.",
    )
    seed: Optional[int] = Field(42, description="Random seed")


class GradeAllRequest(BaseModel):
    """POST /grade/all — grade all tasks at once."""
    actions: Optional[Dict[str, List[List[int]]]] = Field(
        None,
        description="Dict mapping task name → list of actions. Omit to auto-grade all tasks.",
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
