"""
Content Recommendation Environment - OpenEnv Specification
============================================================
Simulates a real-world content recommendation scenario where an AI agent
learns to recommend items (articles, videos, songs) to users.

Observation Space:
  - user_id: int (0-999)
  - session_id: int (unique per episode)
  - user_history: List[int] (past 5 item IDs user interacted with)
  - user_preferences: Dict[str, float] (genre preferences: pop, tech, sports, etc.)
  - user_satisfaction: float (0-1, cumulative satisfaction in session)
  - available_items: List[ContentItem] (500 items with embeddings, metadata)
  - session_step: int (step number in current session, 1-10)

Action Space:
  - List[int] (5 item IDs to recommend, 0-499)

Reward Signals:
  Task 1 (Easy): CTR only
    reward = sum(clicked items) / 5
  
  Task 2 (Medium): CTR + diversity penalty
    reward = CTR - 0.2 * content_diversity_penalty
  
  Task 3 (Hard): Long-term engagement + churn prevention
    reward = engagement - 0.3 * churn_risk_penalty
"""

import asyncio
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
from datetime import datetime


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

class TaskDifficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class ContentItem:
    """Represents a single piece of content (article, video, song, etc.)"""
    item_id: int
    title: str
    genre: str  # pop, tech, sports, entertainment, news
    embedding: List[float]  # 16-dim embedding for similarity
    popularity: float  # base popularity (0-1)
    freshness: int  # days old (affects novelty preference)
    duration: int  # seconds or minutes, affects watch time probability


@dataclass
class UserState:
    """Represents a user's current state"""
    user_id: int
    session_id: int
    interaction_history: List[int]  # last 5 items user clicked
    genre_preferences: Dict[str, float]  # {genre: preference_score}
    satisfaction: float  # cumulative session satisfaction (0-1)
    fatigue: float  # content fatigue (0-1), resets per session
    churn_risk: float  # probability of leaving (0-1)
    session_step: int  # which step in session (1-10)


@dataclass
class ContentRecObservation:
    """What the agent observes at each step"""
    user_state: UserState
    available_items: List[ContentItem]
    last_action: Optional[List[int]] = None  # previous recommendation
    last_clicks: Optional[List[int]] = None  # which items were clicked
    last_reward: float = 0.0


@dataclass
class RecommendationAction:
    """Agent's action: recommend K items"""
    recommended_items: List[int]  # 5 item IDs (0-499)
    
    def __post_init__(self):
        if len(self.recommended_items) != 5:
            raise ValueError(f"Must recommend exactly 5 items, got {len(self.recommended_items)}")
        if len(set(self.recommended_items)) != 5:
            raise ValueError("Duplicate items in recommendation")
        if not all(0 <= item_id < 500 for item_id in self.recommended_items):
            raise ValueError(f"Item IDs must be in [0, 499], got {self.recommended_items}")


@dataclass
class StepResult:
    """Result of env.step()"""
    observation: ContentRecObservation
    reward: float
    done: bool
    info: Dict = None


# ============================================================================
# CONTENT & USER SIMULATORS
# ============================================================================

class ContentCatalog:
    """Generates and manages a catalog of 500 content items"""
    
    GENRES = ["pop", "tech", "sports", "entertainment", "news"]
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.items = self._generate_items()
    
    def _generate_items(self) -> List[ContentItem]:
        items = []
        for item_id in range(500):
            genre = random.choice(self.GENRES)
            # Random 16-dim embedding
            embedding = list(np.random.randn(16))
            popularity = np.random.beta(2, 5)  # skewed towards lower (realistic)
            freshness = random.randint(0, 90)  # 0-90 days old
            duration = random.randint(300, 3600)  # 5 min to 1 hour
            
            items.append(ContentItem(
                item_id=item_id,
                title=f"{genre.capitalize()} Content #{item_id}",
                genre=genre,
                embedding=embedding,
                popularity=popularity,
                freshness=freshness,
                duration=duration,
            ))
        return items
    
    def get_item(self, item_id: int) -> ContentItem:
        return self.items[item_id]
    
    def get_all(self) -> List[ContentItem]:
        return self.items


class UserSimulator:
    """Simulates user behavior: CTR, engagement, churn dynamics"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_user(self, user_id: int) -> UserState:
        """Create a new user with random preferences"""
        # Assign 2-3 favorite genres per user
        all_genres = ContentCatalog.GENRES
        fav_genres = random.sample(all_genres, k=random.randint(2, 3))
        
        genre_prefs = {}
        for genre in all_genres:
            if genre in fav_genres:
                genre_prefs[genre] = random.uniform(0.6, 1.0)
            else:
                genre_prefs[genre] = random.uniform(0.1, 0.4)
        
        return UserState(
            user_id=user_id,
            session_id=random.randint(0, 1000000),
            interaction_history=[],
            genre_preferences=genre_prefs,
            satisfaction=0.5,
            fatigue=0.0,
            churn_risk=0.1,
            session_step=0,
        )
    
    def predict_clicks(
        self,
        user_state: UserState,
        recommended_items: List[int],
        catalog: ContentCatalog,
    ) -> Tuple[List[int], float]:
        """
        Simulate user clicking on recommendations.
        Returns: (clicked_item_ids, engagement_score)
        """
        clicked = []
        total_engagement = 0.0
        
        for item_id in recommended_items:
            item = catalog.get_item(item_id)
            
            # Base CTR from genre preference + popularity
            genre_pref = user_state.genre_preferences.get(item.genre, 0.5)
            base_ctr = 0.3 + 0.3 * genre_pref + 0.2 * item.popularity
            
            # Novelty bonus: user prefers fresh content (inverse of freshness)
            novelty_bonus = 0.1 * (1.0 - min(item.freshness / 90, 1.0))
            
            # Fatigue penalty: repeated genres lower CTR
            fatigue_penalty = user_state.fatigue * 0.15
            
            # Final CTR
            ctr = max(0.0, base_ctr + novelty_bonus - fatigue_penalty)
            
            # User clicks with probability = ctr
            if random.random() < ctr:
                clicked.append(item_id)
                total_engagement += ctr
                
                # Update fatigue: same genre increases fatigue
                user_state.fatigue = min(1.0, user_state.fatigue + 0.15)
            
            total_engagement += ctr * 0.1  # partial engagement even without click
        
        engagement = total_engagement / len(recommended_items)
        return clicked, engagement
    
    def update_user_state(
        self,
        user_state: UserState,
        clicked_items: List[int],
        engagement: float,
        catalog: ContentCatalog,
    ) -> UserState:
        """Update user state based on interaction"""
        # Update interaction history
        user_state.interaction_history.extend(clicked_items)
        user_state.interaction_history = user_state.interaction_history[-5:]
        
        # Update satisfaction
        user_state.satisfaction = 0.7 * user_state.satisfaction + 0.3 * engagement
        
        # Update churn risk (inversely related to satisfaction)
        user_state.churn_risk = max(0.0, 0.15 + 0.7 * (1.0 - user_state.satisfaction))
        
        # Slight fatigue recovery
        user_state.fatigue = max(0.0, user_state.fatigue - 0.05)
        
        return user_state


# ============================================================================
# REWARD FUNCTIONS
# ============================================================================

class RewardFunction:
    """Base class for reward computation"""
    
    def compute(
        self,
        user_state: UserState,
        recommended_items: List[int],
        clicked_items: List[int],
        engagement: float,
        catalog: ContentCatalog,
    ) -> Tuple[float, Dict]:
        raise NotImplementedError


class EasyReward(RewardFunction):
    """Task 1 (Easy): Pure CTR optimization, no complexity"""
    
    def compute(
        self,
        user_state: UserState,
        recommended_items: List[int],
        clicked_items: List[int],
        engagement: float,
        catalog: ContentCatalog,
    ) -> Tuple[float, Dict]:
        # Simple CTR: percentage of recommended items clicked
        ctr = len(clicked_items) / len(recommended_items)
        reward = ctr
        
        return reward, {
            "ctr": ctr,
            "clicks": len(clicked_items),
            "task": "easy",
        }


class MediumReward(RewardFunction):
    """Task 2 (Medium): CTR + content diversity penalty"""
    
    def compute(
        self,
        user_state: UserState,
        recommended_items: List[int],
        clicked_items: List[int],
        engagement: float,
        catalog: ContentCatalog,
    ) -> Tuple[float, Dict]:
        ctr = len(clicked_items) / len(recommended_items)
        
        # Diversity penalty: penalize recommending same genre repeatedly
        genres = [catalog.get_item(item_id).genre for item_id in recommended_items]
        genre_diversity = len(set(genres)) / len(ContentCatalog.GENRES)
        
        # Diversity score: 0.0 if all same genre, 1.0 if all different
        diversity_penalty = (1.0 - genre_diversity) * 0.3
        
        reward = ctr - diversity_penalty
        reward = max(0.0, reward)  # clip to [0, 1]
        
        return reward, {
            "ctr": ctr,
            "diversity_penalty": diversity_penalty,
            "genre_diversity": genre_diversity,
            "clicks": len(clicked_items),
            "task": "medium",
        }


class HardReward(RewardFunction):
    """Task 3 (Hard): Long-term engagement + churn prevention"""
    
    def compute(
        self,
        user_state: UserState,
        recommended_items: List[int],
        clicked_items: List[int],
        engagement: float,
        catalog: ContentCatalog,
    ) -> Tuple[float, Dict]:
        ctr = len(clicked_items) / len(recommended_items)
        
        # Long-term satisfaction bonus
        satisfaction_bonus = user_state.satisfaction * 0.4
        
        # Churn risk penalty: penalize high churn risk
        churn_penalty = user_state.churn_risk * 0.5
        
        # Diversity: encourage exploring new genres to prevent fatigue
        genres = [catalog.get_item(item_id).genre for item_id in recommended_items]
        genre_diversity = len(set(genres)) / len(ContentCatalog.GENRES)
        diversity_bonus = genre_diversity * 0.2
        
        reward = ctr + satisfaction_bonus - churn_penalty + diversity_bonus
        reward = max(0.0, min(1.0, reward))  # clip to [0, 1]
        
        return reward, {
            "ctr": ctr,
            "satisfaction_bonus": satisfaction_bonus,
            "churn_penalty": churn_penalty,
            "diversity_bonus": diversity_bonus,
            "churn_risk": user_state.churn_risk,
            "satisfaction": user_state.satisfaction,
            "task": "hard",
        }


# ============================================================================
# MAIN ENVIRONMENT
# ============================================================================

class ContentRecEnv:
    """
    OpenEnv-compliant content recommendation environment.
    
    Async interface:
      - reset() -> StepResult
      - step(action: RecommendationAction) -> StepResult
      - close() -> None
      - state() -> UserState
    """
    
    def __init__(self, task_difficulty: TaskDifficulty = TaskDifficulty.EASY, seed: int = 42):
        self.task_difficulty = task_difficulty
        self.seed = seed
        self.catalog = ContentCatalog(seed=seed)
        self.user_sim = UserSimulator(seed=seed)
        
        # Select reward function based on difficulty
        if task_difficulty == TaskDifficulty.EASY:
            self.reward_fn = EasyReward()
        elif task_difficulty == TaskDifficulty.MEDIUM:
            self.reward_fn = MediumReward()
        else:  # HARD
            self.reward_fn = HardReward()
        
        self.user_state: Optional[UserState] = None
        self.episode_rewards: List[float] = []
        self.episode_step: int = 0
        self.max_steps = 10  # steps per session
    
    async def reset(self) -> StepResult:
        """Reset environment for new episode"""
        # Generate random user for this episode
        user_id = random.randint(0, 999)
        self.user_state = self.user_sim.generate_user(user_id)
        self.episode_rewards = []
        self.episode_step = 0
        
        # Return initial observation
        obs = ContentRecObservation(
            user_state=self.user_state,
            available_items=self.catalog.get_all(),
            last_action=None,
            last_clicks=None,
            last_reward=0.0,
        )
        
        return StepResult(observation=obs, reward=0.0, done=False)
    
    async def step(self, action: RecommendationAction) -> StepResult:
        """Execute one step: agent recommends, user responds"""
        if self.user_state is None:
            raise RuntimeError("Must call reset() before step()")
        
        # Simulate user clicking
        clicked_items, engagement = self.user_sim.predict_clicks(
            self.user_state,
            action.recommended_items,
            self.catalog,
        )
        
        # Compute reward
        reward, info = self.reward_fn.compute(
            self.user_state,
            action.recommended_items,
            clicked_items,
            engagement,
            self.catalog,
        )
        
        # Update user state for next step
        self.user_state = self.user_sim.update_user_state(
            self.user_state,
            clicked_items,
            engagement,
            self.catalog,
        )
        
        self.episode_step += 1
        self.user_state.session_step = self.episode_step
        self.episode_rewards.append(reward)
        
        # Check if episode is done
        done = self.episode_step >= self.max_steps
        
        # Create observation for next step
        obs = ContentRecObservation(
            user_state=self.user_state,
            available_items=self.catalog.get_all(),
            last_action=action.recommended_items,
            last_clicks=clicked_items,
            last_reward=reward,
        )
        
        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info=info,
        )
    
    async def close(self) -> None:
        """Clean up resources"""
        pass
    
    def state(self) -> UserState:
        """Get current user state"""
        if self.user_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self.user_state
    
    def get_episode_summary(self) -> Dict:
        """Return summary of current episode"""
        return {
            "task": self.task_difficulty.value,
            "steps": self.episode_step,
            "total_reward": sum(self.episode_rewards),
            "avg_reward": sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else 0.0,
            "rewards": self.episode_rewards,
        }


# ============================================================================
# HELPER: Synchronous Wrapper
# ============================================================================

class ContentRecEnvSync:
    """Synchronous wrapper for ContentRecEnv for easier testing"""
    
    def __init__(self, task_difficulty: TaskDifficulty = TaskDifficulty.EASY, seed: int = 42):
        self.env = ContentRecEnv(task_difficulty=task_difficulty, seed=seed)
    
    def reset(self) -> StepResult:
        return asyncio.run(self.env.reset())
    
    def step(self, action: RecommendationAction) -> StepResult:
        return asyncio.run(self.env.step(action))
    
    def close(self) -> None:
        return asyncio.run(self.env.close())
    
    def state(self) -> UserState:
        return self.env.state()
    
    def get_episode_summary(self) -> Dict:
        return self.env.get_episode_summary()


if __name__ == "__main__":
    # Quick test
    env = ContentRecEnvSync(task_difficulty=TaskDifficulty.EASY)
    result = env.reset()
    print(f"Initial observation user_id: {result.observation.user_state.user_id}")
    
    # Take a random action
    action = RecommendationAction(recommended_items=[0, 1, 2, 3, 4])
    result = env.step(action)
    print(f"Step 1 reward: {result.reward:.3f}")
    print(f"Clicked items: {result.observation.last_clicks}")
    
    env.close()
    print("Test passed!")
