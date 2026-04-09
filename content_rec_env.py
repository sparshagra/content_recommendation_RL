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
    EASY   = "easy"
    MEDIUM = "medium"
    HARD   = "hard"


class UserSegment(Enum):
    """
    Behavioural user segments — each has distinct CTR sensitivity and churn dynamics.

    CASUAL  — average engagement, standard fatigue recovery, low churn baseline
    POWER   — high engagement, faster fatigue (discerning), minimal churn risk
    CHURNER — already dissatisfied, narrow preferences, compounding churn risk
    """
    CASUAL  = "casual"
    POWER   = "power"
    CHURNER = "churner"


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
    """Represents a user's current state within a recommendation session."""
    user_id:              int
    session_id:           int
    interaction_history:  List[int]           # last 5 item IDs clicked
    genre_preferences:    Dict[str, float]    # {genre: preference_score [0,1]}
    satisfaction:         float               # session satisfaction (0–1)
    fatigue:              float               # content fatigue (0–1), resets per session
    churn_risk:           float               # probability of leaving (0–1)
    session_step:         int                 # step number in current session
    segment:              str = "casual"      # casual | power | churner
    consecutive_low_sat:  int = 0             # steps with satisfaction < 0.3 (for churn spike)


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
# HELPERS
# ============================================================================

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors. Returns value in [-1, 1]."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _compute_user_pref_vector(user_state: "UserState", catalog: "ContentCatalog") -> List[float]:
    """
    Derive a 16-dim preference vector for the user.
    If the user has interaction history, average the embeddings of clicked items.
    Otherwise, fall back to a genre-weighted average over the full catalog.
    """
    if user_state.interaction_history:
        vecs = [catalog.get_item(iid).embedding for iid in user_state.interaction_history]
        return [sum(v[i] for v in vecs) / len(vecs) for i in range(16)]
    # Fallback: genre-weighted centroid
    total_weight = 0.0
    weighted = [0.0] * 16
    for item in catalog.get_all():
        w = user_state.genre_preferences.get(item.genre, 0.0)
        for i in range(16):
            weighted[i] += w * item.embedding[i]
        total_weight += w
    if total_weight > 0:
        return [v / total_weight for v in weighted]
    return [0.0] * 16


# ============================================================================
# CONTENT & USER SIMULATORS
# ============================================================================

class ContentCatalog:
    """Generates and manages a catalog of 500 content items."""

    GENRES = ["pop", "tech", "sports", "entertainment", "news"]

    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        self.items = self._generate_items()

    def _generate_items(self) -> List[ContentItem]:
        items = []
        for item_id in range(500):
            genre = random.choice(self.GENRES)
            embedding = list(np.random.randn(16))          # 16-dim semantic embedding
            popularity = np.random.beta(2, 5)              # realistically skewed toward low
            freshness = random.randint(0, 90)              # 0-90 days old
            duration = random.randint(300, 3600)           # 5 min – 1 hour
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

    def get_candidates(self, user_state: "UserState", k: int = 50) -> List[ContentItem]:
        """
        Candidate generation stage — returns the top-k most relevant items
        for this user based on genre preference, popularity, and content freshness.

        This mirrors the two-stage retrieve-then-rank pipeline used in production
        recommendation systems (e.g. YouTube, TikTok). The agent ranks from this
        shortlist rather than choosing from all 500 items blindly.
        """
        user_pref_vec = _compute_user_pref_vector(user_state, self)
        scored = []
        for item in self.items:
            genre_score = user_state.genre_preferences.get(item.genre, 0.0)
            freshness_score = 1.0 - min(item.freshness / 90.0, 1.0)
            # Cosine similarity between item embedding and user preference vector
            embed_score = (_cosine_similarity(item.embedding, user_pref_vec) + 1.0) / 2.0  # normalise to [0,1]
            score = 0.4 * genre_score + 0.25 * item.popularity + 0.2 * freshness_score + 0.15 * embed_score
            scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored[:k]]


class UserSimulator:
    """Simulates user behavior: CTR, engagement, churn dynamics"""
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_user(self, user_id: int) -> UserState:
        """
        Create a new user with a randomly assigned behavioural segment and preferences.

        Segments:
          - casual  (50%): standard engagement, moderate preferences
          - power   (30%): highly engaged, broad genre taste, low churn risk
          - churner (20%): narrower preferences, already dissatisfied, high churn risk
        """
        segment_choice = random.choices(
            [UserSegment.CASUAL, UserSegment.POWER, UserSegment.CHURNER],
            weights=[5, 3, 2],
        )[0]
        segment = segment_choice.value

        if segment == "power":
            n_fav            = random.randint(3, 4)
            high_pref_range  = (0.70, 1.00)
            low_pref_range   = (0.20, 0.50)
            init_satisfaction = 0.65
            init_churn       = 0.05
        elif segment == "churner":
            n_fav            = random.randint(1, 2)
            high_pref_range  = (0.50, 0.80)
            low_pref_range   = (0.00, 0.20)
            init_satisfaction = 0.30
            init_churn       = 0.40
        else:  # casual
            n_fav            = random.randint(2, 3)
            high_pref_range  = (0.60, 1.00)
            low_pref_range   = (0.10, 0.40)
            init_satisfaction = 0.50
            init_churn       = 0.10

        all_genres = ContentCatalog.GENRES
        fav_genres = random.sample(all_genres, k=min(n_fav, len(all_genres)))
        genre_prefs = {
            g: random.uniform(*high_pref_range) if g in fav_genres
               else random.uniform(*low_pref_range)
            for g in all_genres
        }

        return UserState(
            user_id=user_id,
            session_id=random.randint(0, 1_000_000),
            interaction_history=[],
            genre_preferences=genre_prefs,
            satisfaction=init_satisfaction,
            fatigue=0.0,
            churn_risk=init_churn,
            session_step=0,
            segment=segment,
            consecutive_low_sat=0,
        )
    
    def predict_clicks(
        self,
        user_state: UserState,
        recommended_items: List[int],
        catalog: ContentCatalog,
    ) -> Tuple[List[int], float]:
        """
        Simulate user click behaviour using a multi-signal CTR model:
          - Genre preference (user profile)
          - Item popularity (base rate)
          - Novelty (content freshness)
          - Semantic similarity (embedding cosine sim with user preference vector)
          - Fatigue penalty (repeated genre exposure)

        Returns: (clicked_item_ids, engagement_score)
        """
        # Derive user's current preference vector from history / genre weights
        user_pref_vec = _compute_user_pref_vector(user_state, catalog)

        clicked = []
        total_engagement = 0.0

        for item_id in recommended_items:
            item = catalog.get_item(item_id)

            # Genre preference signal
            genre_pref = user_state.genre_preferences.get(item.genre, 0.5)

            # Semantic similarity signal — rewards embedding-based personalisation
            raw_sim = _cosine_similarity(item.embedding, user_pref_vec)
            embed_sim = (raw_sim + 1.0) / 2.0  # normalise [-1,1] → [0,1]

            # Novelty bonus: fresh content gets a small lift
            novelty_bonus = 0.08 * (1.0 - min(item.freshness / 90.0, 1.0))

            # Fatigue penalty: stale genre exposure hurts CTR
            fatigue_penalty = user_state.fatigue * 0.15

            # CTR = weighted combination of signals
            base_ctr = (
                0.25 * genre_pref
                + 0.20 * item.popularity
                + 0.15 * embed_sim
                + novelty_bonus
                + 0.25  # base click probability
            )
            ctr = max(0.0, base_ctr - fatigue_penalty)

            if random.random() < ctr:
                clicked.append(item_id)
                total_engagement += ctr
                user_state.fatigue = min(1.0, user_state.fatigue + 0.12)

            total_engagement += ctr * 0.08  # partial engagement even without click

        engagement = total_engagement / len(recommended_items)
        return clicked, engagement
    
    def update_user_state(
        self,
        user_state: UserState,
        clicked_items: List[int],
        engagement: float,
        catalog: ContentCatalog,
    ) -> UserState:
        """
        Update user state dynamics after a step.

        Segment-specific behaviour:
          - power users recover fatigue faster (broad tastes)
          - churner users have compounding churn risk after 3+ consecutive bad steps
        """
        # Interaction history (last 5)
        user_state.interaction_history.extend(clicked_items)
        user_state.interaction_history = user_state.interaction_history[-5:]

        # Satisfaction: exponential moving average
        user_state.satisfaction = 0.7 * user_state.satisfaction + 0.3 * engagement

        # Consecutive low-satisfaction tracking (churner compounding)
        if user_state.satisfaction < 0.3:
            user_state.consecutive_low_sat += 1
        else:
            user_state.consecutive_low_sat = max(0, user_state.consecutive_low_sat - 1)

        # Churn risk: base formula + segment-specific spike
        base_churn = max(0.0, 0.15 + 0.7 * (1.0 - user_state.satisfaction))
        if user_state.segment == "churner" and user_state.consecutive_low_sat >= 3:
            base_churn = min(1.0, base_churn + 0.25)  # compounding churn spike
        user_state.churn_risk = base_churn

        # Fatigue recovery: power users recover faster
        recovery = {"casual": 0.05, "power": 0.08, "churner": 0.03}.get(
            user_state.segment, 0.05
        )
        user_state.fatigue = max(0.0, user_state.fatigue - recovery)

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
    """
    Task 3 (Hard): Long-term user lifetime value with churn prevention.

    Unlike the easy/medium tasks, this reward is NOT clipped at 0 — mild
    negative rewards signal that the agent is actively harming long-term
    retention. This provides a cleaner gradient for learning churn avoidance.

    Reward = CTR + satisfaction_bonus - churn_penalty + diversity_bonus
    Range  = [-0.3, 1.0]  (agent can do worse than zero by inciting churn)
    """

    def compute(
        self,
        user_state: UserState,
        recommended_items: List[int],
        clicked_items: List[int],
        engagement: float,
        catalog: ContentCatalog,
    ) -> Tuple[float, Dict]:
        ctr = len(clicked_items) / len(recommended_items)

        # Long-term value: higher satisfaction means loyal returning user
        satisfaction_bonus = user_state.satisfaction * 0.4

        # Churn penalty: high churn_risk means the user is about to leave
        churn_penalty = user_state.churn_risk * 0.5

        # Genre diversity prevents fatigue and keeps users engaged longer
        genres = [catalog.get_item(iid).genre for iid in recommended_items]
        genre_diversity = len(set(genres)) / len(ContentCatalog.GENRES)
        diversity_bonus = genre_diversity * 0.2

        # Business-interpretable session value: good sessions = retained users
        session_value = ctr * user_state.satisfaction * (1.0 - user_state.churn_risk)

        reward = ctr + satisfaction_bonus - churn_penalty + diversity_bonus
        # Allow mild negatives — this penalises strategies that maximise short-term
        # CTR at the cost of user satisfaction (a real trade-off in production)
        reward = max(-0.3, min(1.0, reward))

        return reward, {
            "ctr": ctr,
            "satisfaction_bonus": satisfaction_bonus,
            "churn_penalty": churn_penalty,
            "diversity_bonus": diversity_bonus,
            "churn_risk": user_state.churn_risk,
            "satisfaction": user_state.satisfaction,
            "session_value": round(session_value, 4),
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
    
    # Variable episode lengths by difficulty — longer episodes expose temporal dynamics
    MAX_STEPS_BY_DIFFICULTY = {
        TaskDifficulty.EASY:   10,
        TaskDifficulty.MEDIUM: 15,
        TaskDifficulty.HARD:   20,
    }
    # Candidate list size — number of pre-filtered items shown per step
    N_CANDIDATES = 50

    def __init__(self, task_difficulty: TaskDifficulty = TaskDifficulty.EASY, seed: int = 42):
        self.task_difficulty = task_difficulty
        self.seed = seed
        self.catalog = ContentCatalog(seed=seed)
        self.user_sim = UserSimulator(seed=seed)

        # Reward function selected by difficulty
        if task_difficulty == TaskDifficulty.EASY:
            self.reward_fn = EasyReward()
        elif task_difficulty == TaskDifficulty.MEDIUM:
            self.reward_fn = MediumReward()
        else:
            self.reward_fn = HardReward()

        self.user_state: Optional[UserState] = None
        self.episode_rewards: List[float] = []
        self.episode_step: int = 0
        self.max_steps: int = self.MAX_STEPS_BY_DIFFICULTY[task_difficulty]
    
    async def reset(self) -> StepResult:
        """
        Reset environment for a new episode.

        Returns the initial observation with a pre-filtered candidate list
        (top-50 items most relevant to this user) rather than the full 500-item
        catalog. This mirrors the realistic two-stage retrieval-then-ranking
        pipeline used in production recommender systems.
        """
        user_id = random.randint(0, 999)
        self.user_state = self.user_sim.generate_user(user_id)
        self.episode_rewards = []
        self.episode_step = 0

        # Candidate generation: top-50 items for this user
        candidates = self.catalog.get_candidates(self.user_state, k=self.N_CANDIDATES)

        obs = ContentRecObservation(
            user_state=self.user_state,
            available_items=candidates,
            last_action=None,
            last_clicks=None,
            last_reward=0.0,
        )
        return StepResult(observation=obs, reward=0.0, done=False)
    
    async def step(self, action: RecommendationAction) -> StepResult:
        """
        Execute one MDP step: agent recommends → user responds → state updates.

        The returned observation contains a freshly generated candidate list
        (top-50 items re-ranked after the user state update) so the agent
        sees how its recommendations shifted the user's preference signal.
        """
        if self.user_state is None:
            raise RuntimeError("Must call reset() before step()")

        # Simulate user clicks with the multi-signal CTR model
        clicked_items, engagement = self.user_sim.predict_clicks(
            self.user_state,
            action.recommended_items,
            self.catalog,
        )

        # Compute reward using the task-appropriate reward function
        reward, info = self.reward_fn.compute(
            self.user_state,
            action.recommended_items,
            clicked_items,
            engagement,
            self.catalog,
        )

        # Add session_value to all tasks (business metric)
        if "session_value" not in info:
            session_value = (
                (len(clicked_items) / len(action.recommended_items))
                * self.user_state.satisfaction
                * (1.0 - self.user_state.churn_risk)
            )
            info["session_value"] = round(session_value, 4)

        # Update user state dynamics
        self.user_state = self.user_sim.update_user_state(
            self.user_state,
            clicked_items,
            engagement,
            self.catalog,
        )

        self.episode_step += 1
        self.user_state.session_step = self.episode_step
        self.episode_rewards.append(reward)

        done = self.episode_step >= self.max_steps

        # Re-generate candidate list with updated user state
        candidates = self.catalog.get_candidates(self.user_state, k=self.N_CANDIDATES)

        obs = ContentRecObservation(
            user_state=self.user_state,
            available_items=candidates,
            last_action=action.recommended_items,
            last_clicks=clicked_items,
            last_reward=reward,
        )
        return StepResult(observation=obs, reward=reward, done=done, info=info)
    
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
