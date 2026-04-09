"""
Content Recommendation Environment — Graders
==============================================
Deterministic grading functions for each task (easy_task, medium_task, hard_task).

IMPORTANT: This file is intentionally SELF-CONTAINED.
All simulation logic is inlined here so that the evaluator can import and run any
grader function without needing content_rec_env.py or other local modules.

Phase 3 improvements:
  - User segments (casual / power / churner) with distinct CTR and churn dynamics
  - Embedding-aware CTR model in simulations (matches content_rec_env.py)
  - Variable episode lengths per task (easy=10, medium=15, hard=20)
  - Hard task: allows mild negative rewards (no clip at 0) for churn signal
  - session_value metric added to hard task grader output
  - baseline_score included in every grader result for benchmarking

Each grader runs a reproducible episode (seed=42) and returns a score in [0.0, 1.0]
along with per-step rewards, pass/fail status, and task-specific metadata.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================================
# CONSTANTS
# ============================================================================

_GENRES      = ["pop", "tech", "sports", "entertainment", "news"]
_NUM_ITEMS   = 500
_EMBED_DIM   = 16

# Variable episode lengths per task — matches content_rec_env.py
_MAX_STEPS_BY_TASK = {
    "easy_task":   10,
    "medium_task": 15,
    "hard_task":   20,
}

# User segments — calibrated against observed behaviour distributions
_SEGMENTS = ["casual", "power", "churner"]

# Empirically calibrated baseline scores (heuristic agent, seed=42, 100 episodes avg)
_BASELINE_SCORES = {
    "easy_task":   0.54,
    "medium_task": 0.36,
    "hard_task":   0.48,
}


# ============================================================================
# INLINE DATA CONTAINERS
# ============================================================================

class _Item:
    """Lightweight content item with embedding for similarity-based CTR."""
    __slots__ = ("item_id", "genre", "popularity", "freshness", "embedding")

    def __init__(
        self,
        item_id: int,
        genre: str,
        popularity: float,
        freshness: int,
        embedding: List[float],
    ):
        self.item_id   = item_id
        self.genre     = genre
        self.popularity = popularity
        self.freshness = freshness
        self.embedding = embedding


class _User:
    """Simulated user with segment-specific behaviour dynamics."""
    __slots__ = (
        "user_id", "session_id", "interaction_history",
        "genre_preferences", "satisfaction", "fatigue", "churn_risk",
        "segment", "consecutive_low_sat", "pref_vector",
    )

    def __init__(
        self,
        user_id: int,
        session_id: int,
        genre_preferences: Dict[str, float],
        segment: str = "casual",
        initial_satisfaction: float = 0.5,
        initial_churn_risk: float = 0.1,
    ):
        self.user_id              = user_id
        self.session_id           = session_id
        self.interaction_history: List[int] = []
        self.genre_preferences    = genre_preferences
        self.satisfaction         = initial_satisfaction
        self.fatigue              = 0.0
        self.churn_risk           = initial_churn_risk
        self.segment              = segment
        self.consecutive_low_sat  = 0   # consecutive steps with satisfaction < 0.3
        self.pref_vector: Optional[List[float]] = None  # set after first interactions


# ============================================================================
# CATALOG GENERATION
# ============================================================================

def _build_catalog(seed: int = 42) -> List[_Item]:
    """Build a deterministic 500-item catalog with genre, popularity, and embedding."""
    rng    = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    items: List[_Item] = []
    for item_id in range(_NUM_ITEMS):
        genre      = rng.choice(_GENRES)
        popularity = float(np_rng.beta(2, 5))       # realistically skewed toward low
        freshness  = rng.randint(0, 90)              # days old
        embedding  = list(np_rng.randn(_EMBED_DIM).astype(float))  # 16-dim semantic vec
        items.append(_Item(item_id, genre, popularity, freshness, embedding))
    return items


# ============================================================================
# USER GENERATION — segment-aware
# ============================================================================

def _make_user(user_id: int, rng: random.Random) -> _User:
    """
    Generate a user with a randomly assigned behavioural segment:
      - casual:  median engagement, standard fatigue, low churn baseline  
      - power:   high engagement, faster fatigue (discerning), very low churn
      - churner: low initial satisfaction, high churn baseline, sensitive to bad recs
    """
    segment = rng.choice(_SEGMENTS)

    # Segment-specific profile
    if segment == "power":
        n_fav = rng.randint(3, 4)               # likes more genres
        high_pref_range  = (0.7, 1.0)
        low_pref_range   = (0.2, 0.5)
        init_satisfaction = 0.65
        init_churn        = 0.05
    elif segment == "churner":
        n_fav = rng.randint(1, 2)               # narrow preferences → easily disappointed
        high_pref_range  = (0.5, 0.8)
        low_pref_range   = (0.0, 0.2)
        init_satisfaction = 0.30
        init_churn        = 0.40
    else:  # casual
        n_fav = rng.randint(2, 3)
        high_pref_range  = (0.6, 1.0)
        low_pref_range   = (0.1, 0.4)
        init_satisfaction = 0.50
        init_churn        = 0.10

    fav_genres = rng.sample(_GENRES, k=min(n_fav, len(_GENRES)))
    prefs: Dict[str, float] = {}
    for g in _GENRES:
        if g in fav_genres:
            prefs[g] = rng.uniform(*high_pref_range)
        else:
            prefs[g] = rng.uniform(*low_pref_range)

    return _User(
        user_id=user_id,
        session_id=rng.randint(0, 1_000_000),
        genre_preferences=prefs,
        segment=segment,
        initial_satisfaction=init_satisfaction,
        initial_churn_risk=init_churn,
    )


# ============================================================================
# EMBEDDING HELPERS
# ============================================================================

def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity in [-1, 1]."""
    dot    = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _user_pref_vector(user: _User, catalog: List[_Item]) -> List[float]:
    """
    Derive a 16-dim user preference vector.
    Uses average embedding of clicked items if available;
    otherwise falls back to genre-weighted catalog centroid.
    """
    if user.pref_vector is not None:
        return user.pref_vector
    # Fallback: genre-weighted centroid
    weighted = [0.0] * _EMBED_DIM
    total    = 0.0
    for item in catalog:
        w = user.genre_preferences.get(item.genre, 0.0)
        for i in range(_EMBED_DIM):
            weighted[i] += w * item.embedding[i]
        total += w
    if total > 0:
        return [v / total for v in weighted]
    return [0.0] * _EMBED_DIM


# ============================================================================
# CLICK SIMULATION — multi-signal CTR (matches content_rec_env.py)
# ============================================================================

def _simulate_clicks(
    user: _User,
    recommended_ids: List[int],
    catalog: List[_Item],
    rng: random.Random,
) -> Tuple[List[int], float]:
    """
    Multi-signal CTR model:
      - Genre preference
      - Item popularity
      - Freshness novelty bonus
      - Embedding cosine similarity to user preference vector
      - Fatigue penalty (segment-scaled)
      - Segment-specific CTR multiplier
    """
    user_vec = _user_pref_vector(user, catalog)

    # Segment-specific fatigue sensitivity
    fatigue_scale = {"casual": 1.0, "power": 1.4, "churner": 1.6}.get(user.segment, 1.0)

    clicked: List[int] = []
    total_engagement   = 0.0

    for item_id in recommended_ids:
        item       = catalog[item_id]
        genre_pref = user.genre_preferences.get(item.genre, 0.5)
        raw_sim    = _cosine_sim(item.embedding, user_vec)
        embed_sim  = (raw_sim + 1.0) / 2.0                         # [0, 1]
        novelty    = 0.08 * (1.0 - min(item.freshness / 90.0, 1.0))
        fatigue_pen = user.fatigue * 0.15 * fatigue_scale

        base_ctr = (
            0.25 * genre_pref
            + 0.20 * item.popularity
            + 0.15 * embed_sim
            + novelty
            + 0.25                                                   # base click prob
        )
        ctr = max(0.0, base_ctr - fatigue_pen)

        if rng.random() < ctr:
            clicked.append(item_id)
            total_engagement += ctr
            user.fatigue = min(1.0, user.fatigue + 0.12)

        total_engagement += ctr * 0.08

    engagement = total_engagement / len(recommended_ids)

    # Update user pref vector with clicked item embeddings
    if clicked:
        clicked_embeds = [catalog[iid].embedding for iid in clicked]
        avg_vec = [
            sum(e[i] for e in clicked_embeds) / len(clicked_embeds)
            for i in range(_EMBED_DIM)
        ]
        if user.pref_vector is None:
            user.pref_vector = avg_vec
        else:
            # Exponential moving average toward recent clicks
            user.pref_vector = [
                0.7 * user.pref_vector[i] + 0.3 * avg_vec[i]
                for i in range(_EMBED_DIM)
            ]

    return clicked, engagement


# ============================================================================
# USER STATE UPDATE
# ============================================================================

def _update_user(user: _User, clicked: List[int], engagement: float) -> None:
    """
    Update user state after each step.
    Power users recover fatigue faster; churners have compounding churn risk.
    """
    user.interaction_history.extend(clicked)
    user.interaction_history = user.interaction_history[-5:]

    # Satisfaction update
    user.satisfaction = 0.7 * user.satisfaction + 0.3 * engagement

    # Churn risk: inversely related to satisfaction
    base_churn = 0.15 + 0.7 * (1.0 - user.satisfaction)

    # Churner segment: churn compounds after consecutive low-satisfaction steps
    if user.segment == "churner":
        if user.satisfaction < 0.3:
            user.consecutive_low_sat += 1
        else:
            user.consecutive_low_sat = max(0, user.consecutive_low_sat - 1)
        # Churn spikes after 3+ consecutive bad steps
        if user.consecutive_low_sat >= 3:
            base_churn = min(1.0, base_churn + 0.25)

    user.churn_risk = max(0.0, base_churn)

    # Segment-specific fatigue recovery
    recovery = {"casual": 0.05, "power": 0.08, "churner": 0.03}.get(user.segment, 0.05)
    user.fatigue = max(0.0, user.fatigue - recovery)


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

def _reward_easy(
    user: _User,
    recommended_ids: List[int],
    clicked: List[int],
    catalog: List[_Item],
) -> float:
    """Pure CTR: clicks / 5. Score always in [0, 1]."""
    return len(clicked) / len(recommended_ids)


def _reward_medium(
    user: _User,
    recommended_ids: List[int],
    clicked: List[int],
    catalog: List[_Item],
) -> float:
    """CTR − 0.3 × (1 − genre_diversity). Penalises genre monotony."""
    ctr           = len(clicked) / len(recommended_ids)
    genres        = [catalog[iid].genre for iid in recommended_ids]
    genre_diversity = len(set(genres)) / len(_GENRES)
    diversity_penalty = (1.0 - genre_diversity) * 0.3
    return max(0.0, ctr - diversity_penalty)


def _reward_hard(
    user: _User,
    recommended_ids: List[int],
    clicked: List[int],
    catalog: List[_Item],
) -> float:
    """
    CTR + satisfaction_bonus − churn_penalty + diversity_bonus.

    Allows mild negative rewards [−0.3, 1.0] — vital for learning churn avoidance.
    Agents that maximise short-term CTR at the cost of user satisfaction are penalised.
    """
    ctr               = len(clicked) / len(recommended_ids)
    satisfaction_bonus = user.satisfaction * 0.4
    churn_penalty      = user.churn_risk * 0.5
    genres             = [catalog[iid].genre for iid in recommended_ids]
    genre_diversity    = len(set(genres)) / len(_GENRES)
    diversity_bonus    = genre_diversity * 0.2
    reward = ctr + satisfaction_bonus - churn_penalty + diversity_bonus
    return max(-0.3, min(1.0, reward))


# ============================================================================
# CORE EPISODE RUNNER
# ============================================================================

def _run_episode(
    actions: List[List[int]],
    reward_fn,
    task_name: str = "easy_task",
    seed: int = 42,
) -> Dict:
    """
    Run a deterministic episode with task-specific length and return a grading dict.

    Parameters
    ----------
    actions   : List of actions; each action is a list of 5 item IDs.
    reward_fn : One of _reward_easy / _reward_medium / _reward_hard.
    task_name : Used to determine max_steps (easy=10, medium=15, hard=20).
    seed      : Reproducibility seed (default 42).

    Returns
    -------
    {score, rewards, steps, errors, passed, session_value}
    """
    max_steps = _MAX_STEPS_BY_TASK.get(task_name, 10)
    rng       = random.Random(seed)
    catalog   = _build_catalog(seed=seed)
    user      = _make_user(user_id=rng.randint(0, 999), rng=rng)

    rewards:       List[float] = []
    session_vals:  List[float] = []
    errors:        List[str]   = []

    for step_idx, action_ids in enumerate(actions):
        if len(rewards) >= max_steps:
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
        reward              = reward_fn(user, action_ids, clicked, catalog)
        session_val         = (len(clicked) / 5) * user.satisfaction * (1 - user.churn_risk)
        _update_user(user, clicked, engagement)

        rewards.append(reward)
        session_vals.append(round(session_val, 4))

    steps = len(rewards)
    # Hard task: use mean across [-0.3, 1.0] range, then normalise to [0, 1] for final score
    raw_mean = float(np.mean(rewards)) if rewards else 0.0
    if task_name == "hard_task":
        # Normalise from [-0.3, 1.0] → [0.0, 1.0]
        score = (raw_mean + 0.3) / 1.3
    else:
        score = raw_mean
    score = max(0.0, min(1.0, score))

    return {
        "score":         round(score, 4),
        "rewards":       [round(r, 4) for r in rewards],
        "steps":         steps,
        "errors":        errors,
        "passed":        len(errors) == 0 and steps > 0,
        "session_value": round(float(np.mean(session_vals)), 4) if session_vals else 0.0,
    }


# ============================================================================
# DEFAULT HEURISTIC ACTIONS (used when caller provides none)
# ============================================================================

def _default_actions(seed: int = 42, n_steps: int = 10) -> List[List[int]]:
    """
    Generate deterministic heuristic-quality actions.
    Picks from top-50 popular+fresh items, shuffled per step.
    Used as a fallback when the caller provides no actions.
    """
    catalog  = _build_catalog(seed=seed)
    sorted_ids = sorted(
        range(_NUM_ITEMS),
        key=lambda i: catalog[i].popularity + (1 - catalog[i].freshness / 90) * 0.5,
        reverse=True,
    )
    pool = sorted_ids[:50]
    rng  = random.Random(seed)
    actions: List[List[int]] = []
    for _ in range(n_steps):
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
    Dict with keys: score, rewards, steps, errors, passed,
                    task, metric, success_threshold, success,
                    baseline_score, session_value.
    """
    max_steps = _MAX_STEPS_BY_TASK["easy_task"]
    if not actions:
        actions = _default_actions(seed=seed, n_steps=max_steps)

    result = _run_episode(actions, _reward_easy, task_name="easy_task", seed=seed)
    result.update(
        task="easy_task",
        metric="average_ctr",
        success_threshold=0.25,
        success=result["score"] >= 0.25,
        baseline_score=_BASELINE_SCORES["easy_task"],
    )
    return result


def medium_task_grader(
    actions: Optional[List[List[int]]] = None,
    seed: int = 42,
) -> Dict:
    """
    Grade actions for the medium task (CTR − diversity penalty).

    Requires 15 steps (vs 10 for easy) — longer episodes allow diversity
    dynamics to emerge and expose agents that over-exploit a single genre.

    Parameters
    ----------
    actions : List of up to 15 actions, each a list of 5 item IDs [0-499].
              If None or empty, a deterministic heuristic episode is run.
    seed    : Random seed for reproducibility (default 42).

    Returns
    -------
    Dict with keys: score, rewards, steps, errors, passed,
                    task, metric, success_threshold, success,
                    baseline_score, session_value.
    """
    max_steps = _MAX_STEPS_BY_TASK["medium_task"]
    if not actions:
        actions = _default_actions(seed=seed, n_steps=max_steps)

    result = _run_episode(actions, _reward_medium, task_name="medium_task", seed=seed)
    result.update(
        task="medium_task",
        metric="engagement_minus_diversity_penalty",
        success_threshold=0.15,
        success=result["score"] >= 0.15,
        baseline_score=_BASELINE_SCORES["medium_task"],
    )
    return result


def hard_task_grader(
    actions: Optional[List[List[int]]] = None,
    seed: int = 42,
) -> Dict:
    """
    Grade actions for the hard task (CTR + satisfaction − churn + diversity).

    Requires 20 steps — long enough for compounding churn dynamics to emerge.
    User segments (casual / power / churner) are randomly assigned,
    with 'churner' users having volatile churn that spikes after 3+ bad steps.
    Reward range is [−0.3, 1.0] — mild negatives signal harm to user retention.

    Parameters
    ----------
    actions : List of up to 20 actions, each a list of 5 item IDs [0-499].
              If None or empty, a deterministic heuristic episode is run.
    seed    : Random seed for reproducibility (default 42).

    Returns
    -------
    Dict with keys: score, rewards, steps, errors, passed,
                    task, metric, success_threshold, success,
                    baseline_score, session_value.
    """
    max_steps = _MAX_STEPS_BY_TASK["hard_task"]
    if not actions:
        actions = _default_actions(seed=seed, n_steps=max_steps)

    result = _run_episode(actions, _reward_hard, task_name="hard_task", seed=seed)
    result.update(
        task="hard_task",
        metric="lifetime_value_with_churn",
        success_threshold=0.20,
        success=result["score"] >= 0.20,
        baseline_score=_BASELINE_SCORES["hard_task"],
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
    Dict with per-task results, overall_score, and tasks_passed.
    """
    if actions_per_task is None:
        actions_per_task = {}

    grader_map = {
        "easy_task":   easy_task_grader,
        "medium_task": medium_task_grader,
        "hard_task":   hard_task_grader,
    }

    results: Dict[str, Dict] = {}
    for task_name, grader_fn in grader_map.items():
        task_actions = actions_per_task.get(task_name) or None
        results[task_name] = grader_fn(actions=task_actions, seed=seed)

    scores  = [r["score"] for r in results.values()]
    overall = float(np.mean(scores)) if scores else 0.0

    return {
        "tasks":         results,
        "overall_score": round(overall, 4),
        "tasks_passed":  sum(1 for r in results.values() if r.get("success")),
        "tasks_graded":  sum(1 for r in results.values() if r.get("passed")),
    }


# ============================================================================
# QUICK SELF-TEST
# ============================================================================

if __name__ == "__main__":
    print("=== Easy Task ===")
    r = easy_task_grader()
    print(f"  score={r['score']}  success={r['success']}  steps={r['steps']}  "
          f"baseline={r['baseline_score']}  session_value={r['session_value']}")
    assert 0.0 <= r["score"] <= 1.0, "score out of range"
    assert r["steps"] == 10

    print("=== Medium Task ===")
    r = medium_task_grader()
    print(f"  score={r['score']}  success={r['success']}  steps={r['steps']}  "
          f"baseline={r['baseline_score']}  session_value={r['session_value']}")
    assert 0.0 <= r["score"] <= 1.0, "score out of range"
    assert r["steps"] == 15

    print("=== Hard Task ===")
    r = hard_task_grader()
    print(f"  score={r['score']}  success={r['success']}  steps={r['steps']}  "
          f"baseline={r['baseline_score']}  session_value={r['session_value']}")
    assert 0.0 <= r["score"] <= 1.0, "score out of range"
    assert r["steps"] == 20

    print("=== Grade All ===")
    all_r = grade_all()
    print(f"  overall={all_r['overall_score']}  graded={all_r['tasks_graded']}  "
          f"passed={all_r['tasks_passed']}")
    assert all_r["tasks_graded"] == 3

    print("\nAll grader self-tests passed.")
