"""
Content Recommendation Inference Script
========================================
Baseline agent using LLM-guided recommendations via OpenAI-compatible API.
Falls back to heuristic (popularity + user preference) if LLM is unavailable.

STDOUT FORMAT: Follows OpenEnv specification
  [START] task=<task_name> env=content-rec model=<model_name>
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

# Required env vars — defaults only on API_BASE_URL and MODEL_NAME (NOT HF_TOKEN)
API_BASE_URL    = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME      = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN        = os.getenv("HF_TOKEN")          # NO default — must be set externally
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")    # NO default — optional fallback
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") # Optional — for from_docker_image()
TASK_NAME       = os.getenv("TASK_NAME", "easy")
BENCHMARK       = os.getenv("BENCHMARK", "content-rec")
MAX_STEPS       = 10  # steps per episode

# Resolve API key: HF_TOKEN > OPENAI_API_KEY > placeholder (avoids client crash)
_api_key = HF_TOKEN or OPENAI_API_KEY or "no-key-set"

# Initialize OpenAI-compatible client pointing to HF Inference / OpenAI endpoint
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=_api_key,
)

# ============================================================================
# LOGGING HELPERS
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    """Emit [STEP] line"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================================
# HEURISTIC RECOMMENDATION AGENT (used as fallback)
# ============================================================================

class HeuristicRecommender:
    """
    Fallback recommendation strategy:
    score(item) = 0.5 * popularity + 0.3 * genre_preference + 0.2 * freshness_bonus
    Penalties for repeated genres and recent recommendations.
    """

    def __init__(self, catalog: ContentCatalog):
        self.catalog = catalog

    def recommend(self, user_state, last_recommendations: Optional[List[int]] = None) -> List[int]:
        items = self.catalog.get_all()
        scores = []

        for item in items:
            popularity_score = item.popularity
            genre_pref = user_state.genre_preferences.get(item.genre, 0.5)
            freshness_bonus = 1.0 - min(item.freshness / 90.0, 1.0)

            score = (
                0.5 * popularity_score + 0.3 * genre_pref + 0.2 * freshness_bonus
            )

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
# LLM RECOMMENDATION AGENT (primary, with heuristic fallback)
# ============================================================================

class LLMRecommender:
    """
    Primary recommender that uses an LLM via OpenAI-compatible API.
    Constructs a prompt from user state and asks the model to pick 5 item IDs.
    Falls back to HeuristicRecommender on any failure.
    """

    def __init__(self, catalog: ContentCatalog):
        self.catalog = catalog
        self.fallback = HeuristicRecommender(catalog)

    def _build_prompt(self, user_state, last_recommendations: Optional[List[int]]) -> str:
        """Build a concise prompt for the LLM."""
        prefs = ", ".join(
            f"{g}={v:.2f}" for g, v in user_state.genre_preferences.items()
        )
        history = user_state.interaction_history if user_state.interaction_history else ["none"]
        last = last_recommendations if last_recommendations else ["none"]

        # Give LLM a representative sample of items (top 50 by popularity) to reason about
        items = sorted(self.catalog.get_all(), key=lambda x: x.popularity, reverse=True)[:50]
        item_list = "\n".join(
            f"  id={it.item_id} genre={it.genre} popularity={it.popularity:.2f} freshness={it.freshness}d"
            for it in items
        )

        prompt = f"""You are a content recommendation agent. Choose the best 5 item IDs to recommend.

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
        return prompt

    def _parse_response(self, text: str) -> Optional[List[int]]:
        """Extract a list of 5 valid item IDs from LLM response text."""
        # Try to find a JSON array in the response
        matches = re.findall(r'\[[\d,\s]+\]', text)
        for m in matches:
            try:
                ids = json.loads(m)
                ids = [int(x) for x in ids]
                if (
                    len(ids) == 5
                    and len(set(ids)) == 5
                    and all(0 <= i <= 499 for i in ids)
                ):
                    return ids
            except (json.JSONDecodeError, ValueError):
                continue
        return None

    def recommend(self, user_state, last_recommendations: Optional[List[int]] = None) -> List[int]:
        """Try LLM first; fall back to heuristic on any error."""
        try:
            prompt = self._build_prompt(user_state, last_recommendations)
            response = client.chat.completions.create(
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
            ids = self._parse_response(text)
            if ids is not None:
                return ids
            # Parsing failed → fallback
            return self.fallback.recommend(user_state, last_recommendations)
        except Exception:
            # LLM unavailable or quota error → fallback silently
            return self.fallback.recommend(user_state, last_recommendations)


# ============================================================================
# MAIN INFERENCE LOOP
# ============================================================================

def run_episode(
    env: ContentRecEnvSync,
    recommender: LLMRecommender,
    task_name: str,
    seed: int = 42,
) -> Dict:
    """Run one episode with the LLM agent (heuristic fallback inside)."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()
        user_state = result.observation.user_state
        last_action = None

        rewards: List[float] = []
        steps_taken = 0
        error_msg = None

        for step_num in range(1, MAX_STEPS + 1):
            if result.done:
                break

            try:
                recommendations = recommender.recommend(user_state, last_action)
                action = RecommendationAction(recommended_items=recommendations)
            except Exception as e:
                error_msg = str(e)
                log_step(
                    step=step_num,
                    action="error",
                    reward=0.0,
                    done=True,
                    error=error_msg,
                )
                steps_taken = step_num
                break

            result = env.step(action)
            reward = result.reward
            done = result.done
            user_state = result.observation.user_state

            rewards.append(reward)
            steps_taken = step_num
            last_action = recommendations

            action_str = f"recommend({recommendations})"
            log_step(step=step_num, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        score = avg_reward
        success = score >= 0.1

    except Exception as e:
        error_msg = str(e)
        print(f"[DEBUG] Episode error: {error_msg}", flush=True)
        rewards = []
        steps_taken = 0
        score = 0.0
        success = False

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task": task_name,
        "steps": steps_taken,
        "score": score,
        "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "total_reward": sum(rewards),
        "num_rewards": len(rewards),
    }


def main():
    """Main entry point"""
    task_str = TASK_NAME.lower()
    if task_str in ["easy", "easy_task"]:
        task_difficulty = TaskDifficulty.EASY
    elif task_str in ["medium", "medium_task"]:
        task_difficulty = TaskDifficulty.MEDIUM
    else:
        task_difficulty = TaskDifficulty.HARD

    env = ContentRecEnvSync(task_difficulty=task_difficulty, seed=42)
    catalog = ContentCatalog(seed=42)
    recommender = LLMRecommender(catalog)

    result = run_episode(env, recommender, task_name=task_str, seed=42)
    print(f"[DEBUG] Episode result: {json.dumps(result, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
