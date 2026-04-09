"""
Content Recommendation Inference Script
========================================
HTTP-based agent using the OpenEnv server API + LLM recommendations.
Mirrors the advaya/OpenEnv evaluator expected pattern exactly:
  1. GET  /health      — wait for server
  2. POST /reset       — start episode, get session_id
  3. POST /step (x N)  — step through episode
  4. GET  /grader      — score the episode via server grader
  5. Log  [START] / [STEP] / [END] per OpenEnv spec

STDOUT FORMAT:
  [START] task=<task_name> env=<env_name> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   task=<task> success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>
"""

import os
import sys
import time
import json
import re
import requests
from typing import List, Optional, Dict

from openai import OpenAI

# Keep ContentCatalog import for building LLM/heuristic prompts locally
from content_rec_env import ContentCatalog

# ============================================================================
# CONFIGURATION — reads required hackathon env vars
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "HuggingFaceH4/zephyr-7b-beta")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = HF_TOKEN or os.getenv("OPENAI_API_KEY") or "no-key-set"

# ENV_API_URL is injected by the OpenEnv evaluator
ENV_API_URL  = os.getenv("ENV_API_URL", "http://localhost:7860")

TASK_NAME    = os.getenv("TASK_NAME", "all")
BENCHMARK    = os.getenv("BENCHMARK", "content-rec")

MAX_STEPS             = 10
MAX_TOTAL_REWARD      = 10.0
SUCCESS_SCORE_THRESHOLD = 0.10


# ============================================================================
# LOGGING HELPERS — must match exact format the evaluator expects
# ============================================================================

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================================
# USER STATE PROXY — adapts HTTP observation dict to an object for recommenders
# ============================================================================

class _UserStateProxy:
    """Wraps the observation dict returned by /reset and /step into an object."""
    def __init__(self, obs_data: dict):
        us = obs_data.get("user_state", obs_data)
        self.user_id             = us.get("user_id", 0)
        self.session_id          = us.get("session_id", 0)
        self.interaction_history = us.get("interaction_history", [])
        self.genre_preferences   = us.get("genre_preferences", {})
        self.satisfaction        = float(us.get("satisfaction", 0.5))
        self.fatigue             = float(us.get("fatigue", 0.0))
        self.churn_risk          = float(us.get("churn_risk", 0.1))
        self.session_step        = us.get("session_step", 0)


# ============================================================================
# HEURISTIC RECOMMENDATION AGENT
# ============================================================================

class HeuristicRecommender:
    """
    Fallback recommendation strategy:
    score(item) = 0.5 * popularity + 0.3 * genre_preference + 0.2 * freshness_bonus
    """

    def __init__(self, catalog: ContentCatalog):
        self.catalog = catalog

    def recommend(
        self, user_state: _UserStateProxy, last_recommendations: Optional[List[int]] = None
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
        self.client   = client
        self.catalog  = catalog
        self.fallback = HeuristicRecommender(catalog)

    def _build_prompt(self, user_state: _UserStateProxy, last_recommendations: Optional[List[int]]) -> str:
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
        self, user_state: _UserStateProxy, last_recommendations: Optional[List[int]] = None
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
# HELPERS
# ============================================================================

def _canonical_task_name(task_name: str) -> str:
    """Normalise to exactly match openenv.yaml task IDs."""
    t = task_name.lower()
    if t in ("easy", "easy_task"):
        return "easy_task"
    if t in ("medium", "medium_task"):
        return "medium_task"
    return "hard_task"


def _wait_for_server(max_retries: int = 20, delay: int = 3) -> bool:
    """Wait until the FastAPI server is live."""
    print(f"# Waiting for environment at {ENV_API_URL}...", flush=True)
    for i in range(max_retries):
        try:
            resp = requests.get(f"{ENV_API_URL}/health", timeout=3)
            if resp.ok:
                print(f"# Environment is LIVE at {ENV_API_URL}", flush=True)
                return True
        except Exception:
            pass
        print(f"# Waiting... ({i + 1}/{max_retries})", flush=True)
        time.sleep(delay)
    print(f"# WARNING: Server not ready after {max_retries} retries, proceeding anyway.", flush=True)
    return False


# ============================================================================
# SINGLE-TASK EPISODE RUNNER — HTTP-based (mirrors advaya pattern)
# ============================================================================

def run_single_task(task_name: str, client: OpenAI) -> Dict:
    """
    Run one full episode for a single task via HTTP API calls:
      POST /reset → loop POST /step → GET /grader
    Emits [START] / [STEP] / [END] per OpenEnv spec.
    """
    canonical   = _canonical_task_name(task_name)
    # server expects short task name: "easy" / "medium" / "hard"
    server_task = canonical.replace("_task", "")

    catalog     = ContentCatalog(seed=42)
    recommender = LLMRecommender(client, catalog)

    rewards:       List[float] = []
    steps_taken:   int         = 0
    score:         float       = 0.0
    success:       bool        = False
    session_id:    Optional[str] = None

    log_start(task=canonical, env=BENCHMARK, model=MODEL_NAME)

    try:
        # ── 1. Reset ──────────────────────────────────────────────────────
        reset_res = requests.post(
            f"{ENV_API_URL}/reset",
            json={"task": server_task, "seed": 42},
            timeout=30,
        )
        reset_res.raise_for_status()

        data       = reset_res.json()
        session_id = data.get("session_id", "default")
        obs_data   = data.get("observation", {})
        user_state = _UserStateProxy(obs_data)

        last_recommendations: Optional[List[int]] = None
        done = False

        # ── 2. Step loop ──────────────────────────────────────────────────
        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get action from LLM (or heuristic fallback)
            try:
                recommendations = recommender.recommend(user_state, last_recommendations)
            except Exception as e:
                log_step(step=step, action="error", reward=0.0, done=True, error=str(e))
                steps_taken = step
                break

            # Step via HTTP — pass session_id so server tracks actions for grader
            step_res = requests.post(
                f"{ENV_API_URL}/step",
                json={"recommended_items": recommendations, "session_id": session_id},
                timeout=15,
            )

            if step_res.status_code != 200:
                err = f"HTTP_{step_res.status_code}"
                log_step(step=step, action=str(recommendations), reward=0.0, done=True, error=err)
                steps_taken = step
                break

            step_data  = step_res.json()
            reward     = float(step_data.get("reward", 0.0))
            done       = bool(step_data.get("done", False))
            obs_data   = step_data.get("observation", obs_data)
            user_state = _UserStateProxy(obs_data)

            rewards.append(reward)
            steps_taken        = step
            last_recommendations = recommendations

            action_str = json.dumps(recommendations)
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

        # ── 3. Call grader via HTTP ───────────────────────────────────────
        # This is the critical call the evaluator uses to detect "has grader"
        try:
            grader_res = requests.get(
                f"{ENV_API_URL}/grader",
                params={"session_id": session_id, "task": canonical},
                timeout=15,
            )
            if grader_res.ok:
                grader_data = grader_res.json()
                score = float(grader_data.get("score", 0.0))
                print(f"[DEBUG] Grader score for {canonical}: {score:.3f}", flush=True)
            else:
                print(f"[DEBUG] Grader returned HTTP {grader_res.status_code}", flush=True)
                score = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.0
        except Exception as e:
            print(f"[DEBUG] Grader call failed: {e}", flush=True)
            score = sum(rewards) / MAX_TOTAL_REWARD if rewards else 0.0

        score   = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as e:
        print(f"[DEBUG] Episode error for {canonical}: {e}", flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(task=canonical, success=success, steps=steps_taken, score=score, rewards=rewards)

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

    # Wait for the FastAPI server to be ready before running tasks
    _wait_for_server()

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
