"""
Content Recommendation Inference Script
========================================
Baseline agent using heuristic recommendations with optional LLM guidance
via OpenAI-compatible API.

STDOUT FORMAT: Follows OpenEnv specification strictly.
  [START] task=<task_name> env=<env_name> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import asyncio
import os
import sys
import json
import re
from typing import List, Optional, Dict

from openai import OpenAI

# Import environment
from content_rec_env import (
    ContentRecEnv,
    ContentRecEnvSync,
    RecommendationAction,
    TaskDifficulty,
    ContentCatalog,
)

# ============================================================================
# ENVIRONMENT CONFIGURATION — reads required hackathon env vars
# ============================================================================

API_BASE_URL    = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME      = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN        = os.getenv("HF_TOKEN")           # Must be set externally
API_KEY         = HF_TOKEN or os.getenv("OPENAI_API_KEY") or "no-key-set"

TASK_NAME       = os.getenv("TASK_NAME", "all")
BENCHMARK       = os.getenv("BENCHMARK", "content-rec")

MAX_STEPS             = 10    # steps per episode
MAX_TOTAL_REWARD      = 10.0  # max possible cumulative reward (10 steps × max 1.0)
SUCCESS_SCORE_THRESHOLD = 0.10  # normalized score threshold to call success


# ============================================================================
# LOGGING HELPERS  — must match exact format the evaluator expects
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line per OpenEnv spec."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    """Emit [STEP] line per OpenEnv spec."""
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line per OpenEnv spec."""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================================
# HEURISTIC RECOMMENDATION AGENT
# ============================================================================

class HeuristicRecommender:
    """
    Fallback recommendation strategy:
    score(item) = 0.5 * popularity + 0.3 * genre_preference + 0.2 * freshness_bonus
    Penalties for repeated genres and recent recommendations.
    """

    def __init__(self, catalog: ContentCatalog):
        self.catalog = catalog

    def recommend(
        self, user_state, last_recommendations: Optional[List[int]] = None
    ) -> List[int]:
        items = self.catalog.get_all()
        scores = []

        for item in items:
            popularity_score = item.popularity
            genre_pref = user_state.genre_preferences.get(item.genre, 0.5)
            freshness_bonus = 1.0 - min(item.freshness / 90.0, 1.0)

            score = 0.5 * popularity_score + 0.3 * genre_pref + 0.2 * freshness_bonus

            if user_state.interaction_history:
                recent_genres = [
                    self.catalog.get_item(item_id).genre
                    for item_id in user_state.interaction_history[-3:]
                ]
                if item.genre in recent_genres:
                    score *= 0.7

            if last_recommendations and item.item_id in last_recommendations:
                score *= 0.5

            scores.append((item.item_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in scores[:5]]


# ============================================================================
# LLM RECOMMENDATION AGENT (primary, heuristic fallback)
# ============================================================================

class LLMRecommender:
    """
    Primary recommender: uses OpenAI-compatible API to pick 5 item IDs.
    Falls back to HeuristicRecommender on any failure.
    """

    def __init__(self, client: OpenAI, catalog: ContentCatalog):
        self.client = client
        self.catalog = catalog
        self.fallback = HeuristicRecommender(catalog)

    def _build_prompt(self, user_state, last_recommendations: Optional[List[int]]) -> str:
        prefs = ", ".join(
            f"{g}={v:.2f}" for g, v in user_state.genre_preferences.items()
        )
        history = user_state.interaction_history or ["none"]
        last    = last_recommendations or ["none"]

        items = sorted(self.catalog.get_all(), key=lambda x: x.popularity, reverse=True)[:50]
        item_list = "\n".join(
            f"  id={it.item_id} genre={it.genre} popularity={it.popularity:.2f} freshness={it.freshness}d"
            for it in items
        )

        return f"""You are a content recommendation agent. Choose the best 5 item IDs to recommend.

User profile:
- Genre preferences (0-1 scale): {prefs}
- Recently clicked items: {history}
- Satisfaction: {user_state.satisfaction:.2f}
- Fatigue: {user_state.fatigue:.2f}
- Churn risk: {user_state.churn_risk:.2f}
- Last recommendations: {last}

Top available items (id, genre, popularity, freshness):
{item_list}

Rules:
- Select exactly 5 unique item IDs from [0, 499]
- Prefer genres the user likes (high preference score)
- Prefer fresh content (low freshness days)
- Avoid repeating last recommendations
- Diversify genres to prevent fatigue

Respond with ONLY a JSON array of 5 integers, e.g.: [12, 45, 99, 201, 387]
"""

    def _parse_response(self, text: str) -> Optional[List[int]]:
        matches = re.findall(r'\[[\d,\s]+\]', text)
        for m in matches:
            try:
                ids = [int(x) for x in json.loads(m)]
                if len(ids) == 5 and len(set(ids)) == 5 and all(0 <= i <= 499 for i in ids):
                    return ids
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    def recommend(
        self, user_state, last_recommendations: Optional[List[int]] = None
    ) -> List[int]:
        try:
            prompt = self._build_prompt(user_state, last_recommendations)
            response = self.client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a smart content recommendation agent. Follow instructions exactly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=64,
                temperature=0.3,
            )
            text = response.choices[0].message.content.strip()
            ids  = self._parse_response(text)
            if ids is not None:
                return ids
        except Exception as exc:
            print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return self.fallback.recommend(user_state, last_recommendations)


# ============================================================================
# SINGLE-TASK EPISODE RUNNER  (matches sample inference script pattern)
# ============================================================================

def _difficulty_for_task(task_name: str) -> TaskDifficulty:
    t = task_name.lower()
    if t in ("easy", "easy_task"):
        return TaskDifficulty.EASY
    if t in ("medium", "medium_task"):
        return TaskDifficulty.MEDIUM
    return TaskDifficulty.HARD


def _canonical_task_name(task_name: str) -> str:
    """Normalise task name to exactly match openenv.yaml task names."""
    t = task_name.lower()
    if t in ("easy", "easy_task"):
        return "easy_task"
    if t in ("medium", "medium_task"):
        return "medium_task"
    return "hard_task"


def run_single_task(task_name: str, client: OpenAI) -> Dict:
    """
    Run one full episode for a single task and emit [START]/[STEP]/[END] logs.
    Matches the sample inference script's log structure exactly.
    """
    canonical = _canonical_task_name(task_name)
    difficulty = _difficulty_for_task(canonical)
    catalog    = ContentCatalog(seed=42)
    recommender = LLMRecommender(client, catalog)

    history:  List[str]  = []
    rewards:  List[float] = []
    steps_taken = 0
    score   = 0.0
    success = False

    log_start(task=canonical, env=BENCHMARK, model=MODEL_NAME)

    env = ContentRecEnvSync(task_difficulty=difficulty, seed=42)

    try:
        result     = env.reset()
        user_state = result.observation.user_state
        last_action: Optional[List[int]] = None
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            try:
                recommendations = recommender.recommend(user_state, last_action)
                action = RecommendationAction(recommended_items=recommendations)
            except Exception as e:
                error_msg = str(e)
                log_step(step=step, action="error", reward=0.0, done=True, error=error_msg)
                steps_taken = step
                break

            result      = env.step(action)
            reward      = result.reward
            done        = result.done
            user_state  = result.observation.user_state

            rewards.append(reward)
            steps_taken = step
            last_action = recommendations
            last_reward = reward

            action_str = json.dumps(recommendations)
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            history.append(f"Step {step}: {recommendations!r} -> reward {reward:+.2f}")

            if done:
                break

        score   = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score   = min(max(score, 0.0), 1.0)          # clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", flush=True)
        rewards     = []
        steps_taken = 0
        score       = 0.0
        success     = False

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task":         canonical,
        "steps":        steps_taken,
        "score":        score,
        "avg_reward":   sum(rewards) / len(rewards) if rewards else 0.0,
        "total_reward": sum(rewards),
        "success":      success,
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    task_str = TASK_NAME.strip().lower()

    if task_str == "all":
        # Run all three tasks — each emits its own [START]/[STEP]/[END] block
        all_tasks = ["easy_task", "medium_task", "hard_task"]
        results   = []
        for t in all_tasks:
            result = run_single_task(t, client)
            results.append(result)
            print(f"[DEBUG] {t} result: {json.dumps(result)}", flush=True)

        overall_avg = sum(r["score"] for r in results) / len(results)
        print(f"[DEBUG] Overall average score: {overall_avg:.3f}", flush=True)

    else:
        result = run_single_task(task_str, client)
        print(f"[DEBUG] Episode result: {json.dumps(result)}", flush=True)


if __name__ == "__main__":
    main()
