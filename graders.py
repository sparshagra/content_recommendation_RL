"""
Content Recommendation Environment — Graders
==============================================
Deterministic grading functions for each task.
OpenEnv Phase 2 requires at least 3 tasks with graders.

Each grader runs a full episode using the environment's reward function
and returns a score in [0.0, 1.0].
"""

import asyncio
from typing import Dict, List, Optional, Tuple

from content_rec_env import (
    ContentRecEnv,
    ContentRecEnvSync,
    ContentCatalog,
    RecommendationAction,
    TaskDifficulty,
)


# ============================================================================
# GRADER BASE
# ============================================================================

class TaskGrader:
    """
    Base grader class. Runs a deterministic episode and scores the agent
    based on the environment's built-in reward function.
    """

    def __init__(self, task_difficulty: TaskDifficulty, seed: int = 42):
        self.task_difficulty = task_difficulty
        self.seed = seed

    def grade(self, actions: List[List[int]]) -> Dict:
        """
        Grade a sequence of actions against the environment.
        
        Args:
            actions: List of recommendation actions (each a list of 5 item IDs)
        
        Returns:
            Dict with score (0.0-1.0), rewards list, and pass/fail status
        """
        env = ContentRecEnvSync(
            task_difficulty=self.task_difficulty, seed=self.seed
        )
        result = env.reset()

        rewards = []
        steps = 0
        errors = []

        for i, action_ids in enumerate(actions):
            if result.done:
                break
            try:
                action = RecommendationAction(recommended_items=action_ids)
                result = env.step(action)
                rewards.append(result.reward)
                steps += 1
            except Exception as e:
                errors.append(f"Step {i+1}: {str(e)}")
                break

        env.close()

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))

        return {
            "score": round(score, 4),
            "rewards": [round(r, 4) for r in rewards],
            "steps": steps,
            "errors": errors,
            "passed": len(errors) == 0 and steps > 0,
        }

    async def grade_async(self, actions: List[List[int]]) -> Dict:
        """Async version of grade."""
        env = ContentRecEnv(
            task_difficulty=self.task_difficulty, seed=self.seed
        )
        result = await env.reset()

        rewards = []
        steps = 0
        errors = []

        for i, action_ids in enumerate(actions):
            if result.done:
                break
            try:
                action = RecommendationAction(recommended_items=action_ids)
                result = await env.step(action)
                rewards.append(result.reward)
                steps += 1
            except Exception as e:
                errors.append(f"Step {i+1}: {str(e)}")
                break

        await env.close()

        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))

        return {
            "score": round(score, 4),
            "rewards": [round(r, 4) for r in rewards],
            "steps": steps,
            "errors": errors,
            "passed": len(errors) == 0 and steps > 0,
        }


# ============================================================================
# TASK-SPECIFIC GRADERS
# ============================================================================

class EasyTaskGrader(TaskGrader):
    """
    Grader for easy_task: Pure CTR optimization.
    Scores based on average click-through rate.
    Success threshold: 0.25
    """

    def __init__(self, seed: int = 42):
        super().__init__(TaskDifficulty.EASY, seed=seed)
        self.success_threshold = 0.25

    def grade(self, actions: List[List[int]]) -> Dict:
        result = super().grade(actions)
        result["task"] = "easy_task"
        result["metric"] = "average_ctr"
        result["success_threshold"] = self.success_threshold
        result["success"] = result["score"] >= self.success_threshold
        return result


class MediumTaskGrader(TaskGrader):
    """
    Grader for medium_task: Engagement + diversity.
    Scores based on CTR minus diversity penalty.
    Success threshold: 0.15
    """

    def __init__(self, seed: int = 42):
        super().__init__(TaskDifficulty.MEDIUM, seed=seed)
        self.success_threshold = 0.15

    def grade(self, actions: List[List[int]]) -> Dict:
        result = super().grade(actions)
        result["task"] = "medium_task"
        result["metric"] = "engagement_minus_diversity_penalty"
        result["success_threshold"] = self.success_threshold
        result["success"] = result["score"] >= self.success_threshold
        return result


class HardTaskGrader(TaskGrader):
    """
    Grader for hard_task: Long-term user lifetime value with churn prevention.
    Scores based on combined CTR, satisfaction, churn, and diversity.
    Success threshold: 0.20
    """

    def __init__(self, seed: int = 42):
        super().__init__(TaskDifficulty.HARD, seed=seed)
        self.success_threshold = 0.20

    def grade(self, actions: List[List[int]]) -> Dict:
        result = super().grade(actions)
        result["task"] = "hard_task"
        result["metric"] = "lifetime_value_with_churn"
        result["success_threshold"] = self.success_threshold
        result["success"] = result["score"] >= self.success_threshold
        return result


# ============================================================================
# MODULE-LEVEL GRADER FUNCTIONS (referenced from openenv.yaml)
# ============================================================================

_easy_grader = EasyTaskGrader()
_medium_grader = MediumTaskGrader()
_hard_grader = HardTaskGrader()


def easy_task_grader(actions: List[List[int]], seed: int = 42) -> Dict:
    """Grade actions for the easy task (CTR optimization)."""
    grader = EasyTaskGrader(seed=seed)
    return grader.grade(actions)


def medium_task_grader(actions: List[List[int]], seed: int = 42) -> Dict:
    """Grade actions for the medium task (engagement + diversity)."""
    grader = MediumTaskGrader(seed=seed)
    return grader.grade(actions)


def hard_task_grader(actions: List[List[int]], seed: int = 42) -> Dict:
    """Grade actions for the hard task (lifetime value + churn)."""
    grader = HardTaskGrader(seed=seed)
    return grader.grade(actions)


# ============================================================================
# GRADE ALL TASKS
# ============================================================================

def grade_all(actions_per_task: Dict[str, List[List[int]]], seed: int = 42) -> Dict:
    """
    Grade all tasks at once.
    
    Args:
        actions_per_task: Dict mapping task name to its action sequence
            e.g. {"easy_task": [[1,2,3,4,5], ...], "medium_task": [...], "hard_task": [...]}
        seed: Random seed for reproducibility
    
    Returns:
        Dict with results for each task and an overall score
    """
    graders = {
        "easy_task": easy_task_grader,
        "medium_task": medium_task_grader,
        "hard_task": hard_task_grader,
    }
    
    results = {}
    for task_name, grader_fn in graders.items():
        task_actions = actions_per_task.get(task_name, [])
        if task_actions:
            results[task_name] = grader_fn(task_actions, seed=seed)
        else:
            results[task_name] = {
                "score": 0.0,
                "rewards": [],
                "steps": 0,
                "errors": ["No actions provided"],
                "passed": False,
                "task": task_name,
                "success": False,
            }
    
    # Overall score is the average of all task scores
    scores = [r["score"] for r in results.values()]
    overall = sum(scores) / len(scores) if scores else 0.0
    
    return {
        "tasks": results,
        "overall_score": round(overall, 4),
        "num_tasks_graded": len([r for r in results.values() if r.get("passed")]),
    }


if __name__ == "__main__":
    # Quick test: grade random actions for all tasks
    import random
    random.seed(42)
    
    test_actions = [[random.randint(0, 499) for _ in range(5)] for _ in range(10)]
    
    print("=== Easy Task Grading ===")
    result = easy_task_grader(test_actions)
    print(f"Score: {result['score']}, Success: {result['success']}")
    
    print("\n=== Medium Task Grading ===")
    result = medium_task_grader(test_actions)
    print(f"Score: {result['score']}, Success: {result['success']}")
    
    print("\n=== Hard Task Grading ===")
    result = hard_task_grader(test_actions)
    print(f"Score: {result['score']}, Success: {result['success']}")
    
    print("\n=== Grade All ===")
    all_result = grade_all({
        "easy_task": test_actions,
        "medium_task": test_actions,
        "hard_task": test_actions,
    })
    print(f"Overall Score: {all_result['overall_score']}")
    print(f"Tasks Graded: {all_result['num_tasks_graded']}")
