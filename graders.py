"""
Content Recommendation Environment — Graders
==============================================
Deterministic grading functions for each task (easy_task, medium_task, hard_task).

IMPORTANT: This file is intentionally SELF-CONTAINED.
All simulation logic is inlined here so that the evaluator can import and
run any grader function without needing content_rec_env.py or any other
local module installed in its environment.

Each grader runs a reproducible episode (seed=42) and returns a score in
[0.0, 1.0] along with per-step rewards and pass/fail status.
"""

import random
from typing import Dict, List, Optional, Tuple

# numpy is the only non-stdlib dependency; it is listed in requirements.txt
import numpy as np


# ============================================================================
# INLINE SIMULATION — mirrors content_rec_env.py exactly
# ============================================================================

_GENRES = ["pop", "tech", "sports", "entertainment", "news"]
_NUM_ITEMS = 500
_MAX_STEPS = 10


# ── Data containers ──────────────────────────────────────────────────────────

class _Item:
    __slots__ = ("item_id", "genre", "popularity", "freshness")

    def __init__(self, item_id: int, genre: str, popularity: float, freshness: int):
        self.item_id = item_id
        self.genre = genre
        self.popularity = popularity
        self.freshness = freshness


class _User:
    __slots__ = (
        "user_id", "session_id", "interaction_history",
        "genre_preferences", "satisfaction", "fatigue", "churn_risk",
    )

    def __init__(
        self,
        user_id: int,
        session_id: int,
        genre_preferences: Dict[str, float],
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.interaction_history: List[int] = []
        self.genre_preferences = genre_preferences
        self.satisfaction: float = 0.5
        self.fatigue: float = 0.0
        self.churn_risk: float = 0.1


# ── Catalog ───────────────────────────────────────────────────────────────────

def _build_catalog(seed: int = 42) -> List[_Item]:
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    items: List[_Item] = []
    for item_id in range(_NUM_ITEMS):
        genre = rng.choice(_GENRES)
        popularity = float(np_rng.beta(2, 5))
        freshness = rng.randint(0, 90)
        items.append(_Item(item_id, genre, popularity, freshness))
    return items


# ── User generation ───────────────────────────────────────────────────────────

def _make_user(user_id: int, rng: random.Random) -> _User:
    fav_genres = rng.sample(_GENRES, k=rng.randint(2, 3))
    prefs: Dict[str, float] = {}
    for g in _GENRES:
        if g in fav_genres:
            prefs[g] = rng.uniform(0.6, 1.0)
        else:
            prefs[g] = rng.uniform(0.1, 0.4)
    return _User(
        user_id=user_id,
        session_id=rng.randint(0, 1_000_000),
        genre_preferences=prefs,
    )


# ── Click simulation ──────────────────────────────────────────────────────────

def _simulate_clicks(
    user: _User,
    recommended_ids: List[int],
    catalog: List[_Item],
    rng: random.Random,
) -> Tuple[List[int], float]:
    """Returns (clicked_ids, engagement_score)."""
    clicked: List[int] = []
    total_engagement = 0.0

    for item_id in recommended_ids:
        item = catalog[item_id]
        genre_pref = user.genre_preferences.get(item.genre, 0.5)
        base_ctr = 0.3 + 0.3 * genre_pref + 0.2 * item.popularity
        novelty_bonus = 0.1 * (1.0 - min(item.freshness / 90.0, 1.0))
        fatigue_penalty = user.fatigue * 0.15
        ctr = max(0.0, base_ctr + novelty_bonus - fatigue_penalty)

        if rng.random() < ctr:
            clicked.append(item_id)
            total_engagement += ctr
            user.fatigue = min(1.0, user.fatigue + 0.15)

        total_engagement += ctr * 0.1  # partial engagement without click

    engagement = total_engagement / len(recommended_ids)
    return clicked, engagement


# ── User-state update ─────────────────────────────────────────────────────────

def _update_user(user: _User, clicked: List[int], engagement: float) -> None:
    user.interaction_history.extend(clicked)
    user.interaction_history = user.interaction_history[-5:]
    user.satisfaction = 0.7 * user.satisfaction + 0.3 * engagement
    user.churn_risk = max(0.0, 0.15 + 0.7 * (1.0 - user.satisfaction))
    user.fatigue = max(0.0, user.fatigue - 0.05)


# ── Reward functions ──────────────────────────────────────────────────────────

def _reward_easy(
    user: _User,
    recommended_ids: List[int],
    clicked: List[int],
    catalog: List[_Item],
) -> float:
    """Pure CTR: clicks / 5."""
    return len(clicked) / len(recommended_ids)


def _reward_medium(
    user: _User,
    recommended_ids: List[int],
    clicked: List[int],
    catalog: List[_Item],
) -> float:
    """CTR - 0.3 * (1 - genre_diversity)."""
    ctr = len(clicked) / len(recommended_ids)
    genres = [catalog[iid].genre for iid in recommended_ids]
    genre_diversity = len(set(genres)) / len(_GENRES)
    diversity_penalty = (1.0 - genre_diversity) * 0.3
    return max(0.0, ctr - diversity_penalty)


def _reward_hard(
    user: _User,
    recommended_ids: List[int],
    clicked: List[int],
    catalog: List[_Item],
) -> float:
    """CTR + 0.4*satisfaction - 0.5*churn + 0.2*diversity."""
    ctr = len(clicked) / len(recommended_ids)
    satisfaction_bonus = user.satisfaction * 0.4
    churn_penalty = user.churn_risk * 0.5
    genres = [catalog[iid].genre for iid in recommended_ids]
    genre_diversity = len(set(genres)) / len(_GENRES)
    diversity_bonus = genre_diversity * 0.2
    reward = ctr + satisfaction_bonus - churn_penalty + diversity_bonus
    return max(0.0, min(1.0, reward))


# ── Core runner ───────────────────────────────────────────────────────────────

def _run_episode(
    actions: List[List[int]],
    reward_fn,          # callable(_user, ids, clicked, catalog) -> float
    seed: int = 42,
) -> Dict:
    """
    Run a deterministic episode and return a grading dict.

    Parameters
    ----------
    actions   : List of 10 actions; each action is a list of 5 item IDs.
    reward_fn : One of _reward_easy / _reward_medium / _reward_hard.
    seed      : Reproducibility seed (default 42).

    Returns
    -------
    {score, rewards, steps, errors, passed}
    """
    rng = random.Random(seed)
    catalog = _build_catalog(seed=seed)
    user = _make_user(user_id=rng.randint(0, 999), rng=rng)

    rewards: List[float] = []
    errors: List[str] = []

    for step_idx, action_ids in enumerate(actions):
        if len(rewards) >= _MAX_STEPS:
            break

        # Validate action
        if not isinstance(action_ids, list) or len(action_ids) != 5:
            errors.append(f"Step {step_idx + 1}: action must be a list of 5 integers")
            break
        if len(set(action_ids)) != 5:
            errors.append(f"Step {step_idx + 1}: duplicate item IDs in action")
            break
        if not all(isinstance(i, int) and 0 <= i < _NUM_ITEMS for i in action_ids):
            errors.append(f"Step {step_idx + 1}: item IDs must be integers in [0, 499]")
            break

        clicked, engagement = _simulate_clicks(user, action_ids, catalog, rng)
        reward = reward_fn(user, action_ids, clicked, catalog)
        _update_user(user, clicked, engagement)

        rewards.append(reward)

    steps = len(rewards)
    score = float(np.mean(rewards)) if rewards else 0.0
    score = max(0.0, min(1.0, score))

    return {
        "score": round(score, 4),
        "rewards": [round(r, 4) for r in rewards],
        "steps": steps,
        "errors": errors,
        "passed": len(errors) == 0 and steps > 0,
    }


# ── Default heuristic actions (used when caller provides none) ────────────────

def _default_actions(seed: int = 42) -> List[List[int]]:
    """
    Generate 10 deterministic heuristic-quality actions.
    Picks the 5 most popular items that haven't been used recently, rotated.
    """
    catalog = _build_catalog(seed=seed)
    sorted_ids = sorted(range(_NUM_ITEMS), key=lambda i: catalog[i].popularity, reverse=True)

    actions: List[List[int]] = []
    rng = random.Random(seed)
    pool = sorted_ids[:50]  # top-50 popular items

    for _ in range(_MAX_STEPS):
        rng.shuffle(pool)
        actions.append(pool[:5])

    return actions


# ============================================================================
# PUBLIC GRADER FUNCTIONS — referenced from openenv.yaml
# ============================================================================

def easy_task_grader(
    actions: Optional[List[List[int]]] = None,
    seed: int = 42,
) -> Dict:
    """
    Grade actions for the easy task (pure CTR optimisation).

    Parameters
    ----------
    actions : List of up to 10 actions, each a list of 5 item IDs [0-499].
              If None or empty, a deterministic heuristic episode is run.
    seed    : Random seed for reproducibility (default 42).

    Returns
    -------
    Dict with keys:
      score (float [0,1]), rewards (list[float]), steps (int),
      errors (list[str]), passed (bool),
      task, metric, success_threshold, success.
    """
    if not actions:
        actions = _default_actions(seed=seed)

    result = _run_episode(actions, _reward_easy, seed=seed)
    result.update(
        task="easy_task",
        metric="average_ctr",
        success_threshold=0.25,
        success=result["score"] >= 0.25,
    )
    return result


def medium_task_grader(
    actions: Optional[List[List[int]]] = None,
    seed: int = 42,
) -> Dict:
    """
    Grade actions for the medium task (CTR - diversity penalty).

    Parameters
    ----------
    actions : List of up to 10 actions, each a list of 5 item IDs [0-499].
              If None or empty, a deterministic heuristic episode is run.
    seed    : Random seed for reproducibility (default 42).

    Returns
    -------
    Dict with keys:
      score (float [0,1]), rewards (list[float]), steps (int),
      errors (list[str]), passed (bool),
      task, metric, success_threshold, success.
    """
    if not actions:
        actions = _default_actions(seed=seed)

    result = _run_episode(actions, _reward_medium, seed=seed)
    result.update(
        task="medium_task",
        metric="engagement_minus_diversity_penalty",
        success_threshold=0.15,
        success=result["score"] >= 0.15,
    )
    return result


def hard_task_grader(
    actions: Optional[List[List[int]]] = None,
    seed: int = 42,
) -> Dict:
    """
    Grade actions for the hard task (CTR + satisfaction - churn + diversity).

    Parameters
    ----------
    actions : List of up to 10 actions, each a list of 5 item IDs [0-499].
              If None or empty, a deterministic heuristic episode is run.
    seed    : Random seed for reproducibility (default 42).

    Returns
    -------
    Dict with keys:
      score (float [0,1]), rewards (list[float]), steps (int),
      errors (list[str]), passed (bool),
      task, metric, success_threshold, success.
    """
    if not actions:
        actions = _default_actions(seed=seed)

    result = _run_episode(actions, _reward_hard, seed=seed)
    result.update(
        task="hard_task",
        metric="lifetime_value_with_churn",
        success_threshold=0.20,
        success=result["score"] >= 0.20,
    )
    return result


def grade_all(
    actions_per_task: Optional[Dict[str, List[List[int]]]] = None,
    seed: int = 42,
) -> Dict:
    """
    Grade all three tasks at once.

    Parameters
    ----------
    actions_per_task : Dict mapping task name → list of actions.
                       Missing tasks are auto-graded with heuristic actions.
    seed             : Random seed (default 42).

    Returns
    -------
    Dict with per-task results and an overall_score.
    """
    if actions_per_task is None:
        actions_per_task = {}

    grader_map = {
        "easy_task": easy_task_grader,
        "medium_task": medium_task_grader,
        "hard_task": hard_task_grader,
    }

    results: Dict[str, Dict] = {}
    for task_name, grader_fn in grader_map.items():
        task_actions = actions_per_task.get(task_name) or None
        results[task_name] = grader_fn(actions=task_actions, seed=seed)

    scores = [r["score"] for r in results.values()]
    overall = float(np.mean(scores)) if scores else 0.0

    return {
        "tasks": results,
        "overall_score": round(overall, 4),
        "num_tasks_graded": sum(1 for r in results.values() if r.get("passed")),
    }


# ============================================================================
# QUICK SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=== Easy Task ===")
    r = easy_task_grader()
    print(f"  score={r['score']}  success={r['success']}  steps={r['steps']}")
    assert 0.0 <= r["score"] <= 1.0, "score out of range"

    print("=== Medium Task ===")
    r = medium_task_grader()
    print(f"  score={r['score']}  success={r['success']}  steps={r['steps']}")
    assert 0.0 <= r["score"] <= 1.0, "score out of range"

    print("=== Hard Task ===")
    r = hard_task_grader()
    print(f"  score={r['score']}  success={r['success']}  steps={r['steps']}")
    assert 0.0 <= r["score"] <= 1.0, "score out of range"

    print("=== Grade All ===")
    all_r = grade_all()
    print(f"  overall_score={all_r['overall_score']}  tasks_graded={all_r['num_tasks_graded']}")
    assert all_r["num_tasks_graded"] == 3

    print("\n✅ All grader self-tests passed.")
