# Content Recommendation OpenEnv Environment

A **production-grade OpenEnv environment** simulating real-world content recommendation scenarios. AI agents learn to maximize user engagement while managing diversity, fatigue, and long-term retention.

## 🎯 Overview

This environment models a streaming/social platform where:
- **Users** have preferences, fatigue, and churn risk
- **Content** is characterized by genre, popularity, freshness, and duration
- **Agent** recommends 5 items per session and receives reward signals
- **Three difficulty levels** test different aspects of recommendation (CTR → diversity → churn)

**Real-world relevance:** Netflix, YouTube, TikTok, Spotify all optimize exactly this problem. Agents that perform well here could inform production systems.

---

## 🚀 Quick Start

### Installation

```bash
# Clone or download the environment
git clone <repo-url>
cd content-rec-openenv

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from content_rec_env import ContentRecEnvSync, RecommendationAction, TaskDifficulty

# Initialize environment
env = ContentRecEnvSync(task_difficulty=TaskDifficulty.EASY)

# Reset for new episode
result = env.reset()
user_state = result.observation.user_state

# Take action: recommend 5 items
action = RecommendationAction(recommended_items=[0, 1, 2, 3, 4])
result = env.step(action)

print(f"Reward: {result.reward:.3f}")
print(f"User clicked: {result.observation.last_clicks}")
print(f"Done: {result.done}")

env.close()
```

### Run Baseline Agent

```bash
# Easy task
TASK_NAME=easy python inference.py

# Medium task
TASK_NAME=medium python inference.py

# Hard task
TASK_NAME=hard python inference.py
```

Expected baseline scores:
- **Easy:** ~0.54 (CTR optimization)
- **Medium:** ~0.36 (CTR + diversity)
- **Hard:** ~0.51 (Long-term value)

---

## 📊 Environment Specification

### Observation Space

The agent receives a structured observation at each step:

```python
@dataclass
class ContentRecObservation:
    user_state: UserState          # User preferences, history, state
    available_items: List[ContentItem]  # 500 items with metadata
    last_action: List[int] | None  # Previous recommendations
    last_clicks: List[int] | None  # Items user clicked
    last_reward: float             # Reward from previous step
```

**UserState Components:**
- `user_id`: Unique identifier (0-999)
- `session_id`: Episode identifier
- `interaction_history`: Last 5 clicked items
- `genre_preferences`: Dict mapping genre → preference score (0-1)
- `satisfaction`: Cumulative session satisfaction (0-1)
- `fatigue`: Content fatigue level (0-1, higher = less engagement)
- `churn_risk`: Probability user will leave (0-1)
- `session_step`: Current step (1-10)

**ContentItem Components:**
- `item_id`: Index in catalog (0-499)
- `title`: Descriptive name
- `genre`: One of {pop, tech, sports, entertainment, news}
- `embedding`: 16-dimensional vector for similarity
- `popularity`: Base popularity score (0-1)
- `freshness`: Days old (0-90)
- `duration`: Length in seconds

### Action Space

```python
@dataclass
class RecommendationAction:
    recommended_items: List[int]  # Exactly 5 item IDs from [0, 499]
```

**Constraints:**
- Must recommend exactly 5 items
- No duplicate items
- Item IDs in range [0, 499]

### Reward Functions

#### Task 1: Easy (CTR Optimization)

Pure click-through rate:

```
reward = (# items clicked) / 5
range: [0.0, 1.0]
```

**Difficulty:** Easy — greedy popularity-based recommendations work well
**Baseline:** 0.540
**Success Threshold:** 0.25

---

#### Task 2: Medium (Engagement + Diversity)

CTR with content diversity penalty:

```
genre_diversity = (# unique genres recommended) / 5

reward = CTR - 0.3 * (1.0 - genre_diversity)
range: [0.0, 1.0]
```

**Difficulty:** Medium — requires balancing CTR vs. exploration
**Baseline:** 0.360
**Success Threshold:** 0.15

**Why it matters:** Filter bubbles hurt long-term engagement. Systems must diversify.

---

#### Task 3: Hard (Lifetime Value + Churn Prevention)

Multi-objective optimization with temporal dynamics:

```
satisfaction_bonus = user_satisfaction * 0.4
churn_penalty = user_churn_risk * 0.5
diversity_bonus = genre_diversity * 0.2

reward = CTR + satisfaction_bonus - churn_penalty + diversity_bonus
range: [0.0, 1.0]
```

**Difficulty:** Hard — requires understanding how recommendations affect future state
**Baseline:** 0.506
**Success Threshold:** 0.20

**Why it matters:** Long-term retention is more valuable than short-term clicks.

---

## 🔄 API Reference

### `ContentRecEnv` (Async)

```python
class ContentRecEnv:
    async def reset() -> StepResult
    async def step(action: RecommendationAction) -> StepResult
    async def close() -> None
    def state() -> UserState
    def get_episode_summary() -> Dict
```

**StepResult:**
```python
@dataclass
class StepResult:
    observation: ContentRecObservation
    reward: float
    done: bool
    info: Dict  # Task-specific metrics
```

### `ContentRecEnvSync` (Synchronous Wrapper)

For simpler usage without async:

```python
env = ContentRecEnvSync(
    task_difficulty=TaskDifficulty.EASY,
    seed=42
)

result = env.reset()
result = env.step(action)
env.close()
```

---

## 📈 User Behavior Simulation

### Click Prediction

For each recommended item:

```
base_ctr = 0.3 + 0.3 * genre_preference + 0.2 * item_popularity
novelty_bonus = 0.1 * (1.0 - min(freshness / 90, 1.0))
fatigue_penalty = user_fatigue * 0.15

ctr = max(0.0, base_ctr + novelty_bonus - fatigue_penalty)
P(click) = ctr
```

Users are more likely to click items matching their preferences, fresh content, and popular items, but fatigue reduces engagement.

### State Evolution

After each step, user state updates:

```python
# Satisfaction: moving average of engagement
satisfaction = 0.7 * old_satisfaction + 0.3 * engagement

# Churn risk: inverse of satisfaction
churn_risk = max(0.0, 0.15 + 0.7 * (1.0 - satisfaction))

# Fatigue: accumulates with repeated genres, decays over time
fatigue = min(1.0, fatigue + 0.15 * genre_penalty) - 0.05
```

**Insight:** Recommending the same genres repeatedly increases fatigue, reducing future clicks.

---

## 🏆 Baseline Agent

The `inference.py` script implements a **heuristic recommender:**

```
score(item) = 0.5 * popularity 
            + 0.3 * genre_preference
            + 0.2 * freshness_bonus
            - penalty_if_repeated_genre
            - penalty_if_recent_recommendation
```

**Strategy:**
1. Score all 500 items
2. Apply penalties for diversity and repeats
3. Select top-5 items

**Why it works:** Simple + explainable, but suboptimal for medium/hard tasks.

**How to beat it:**
- **Easy:** Use popularity + preferences (baseline-level strategy)
- **Medium:** Add diversity awareness (plan for different genres each turn)
- **Hard:** Model how recommendations affect user satisfaction (e.g., use history to predict churn)

---

## 🧪 Testing & Evaluation

### Run Single Episode

```python
from content_rec_env import ContentRecEnvSync, RecommendationAction, TaskDifficulty

env = ContentRecEnvSync(task_difficulty=TaskDifficulty.MEDIUM)
result = env.reset()

for step in range(10):
    action = RecommendationAction(recommended_items=[0, 1, 2, 3, 4])
    result = env.step(action)
    print(f"Step {step+1}: reward={result.reward:.3f}, done={result.done}")
    
    if result.done:
        break

summary = env.get_episode_summary()
print(f"Episode summary: {summary}")
env.close()
```

### Reproduce Baseline Scores

```bash
./eval_baseline.sh  # Runs 5 episodes per task, computes average
```

Or manually:

```bash
python inference.py > /tmp/easy.log     && grep "^\\[END\\]" /tmp/easy.log
TASK_NAME=medium python inference.py > /tmp/medium.log && grep "^\\[END\\]" /tmp/medium.log
TASK_NAME=hard python inference.py > /tmp/hard.log     && grep "^\\[END\\]" /tmp/hard.log
```

Expected output:
```
[END] success=true steps=10 score=0.540 rewards=1.00,0.60,0.60,0.60,0.40,0.60,0.40,0.00,0.60,0.60
[END] success=true steps=10 score=0.360 rewards=0.82,0.42,0.42,0.42,0.22,0.36,0.22,0.00,0.36,0.36
[END] success=true steps=10 score=0.506 rewards=1.00,0.70,0.64,0.62,0.38,0.52,0.34,0.00,0.42,0.45
```

---

## 📝 Inference Script Format

The `inference.py` script follows the **OpenEnv standard output format**:

```
[START] task=<task> env=<env> model=<model>
[STEP]  step=<n> action=<action> reward=<r> done=<bool> error=<msg|null>
[END]   success=<bool> steps=<n> score=<score> rewards=<r1,r2,...>
```

**Line Types:**

1. **[START]** — Emitted once at episode begin
   - `task`: Task name (easy, medium, hard)
   - `env`: Environment name (content-rec)
   - `model`: Agent name (heuristic-baseline, etc.)

2. **[STEP]** — Emitted after each env.step()
   - `step`: Step number (1-10)
   - `action`: Action taken (string representation)
   - `reward`: Reward received (2 decimal places)
   - `done`: Episode finished (true/false)
   - `error`: Error message or null

3. **[END]** — Emitted after episode cleanup
   - `success`: Success flag (true if score >= threshold)
   - `steps`: Total steps taken
   - `score`: Final normalized score [0.0, 1.0]
   - `rewards`: Comma-separated rewards per step

**Example:**
```
[START] task=easy env=content-rec model=heuristic-baseline
[STEP] step=1 action=recommend([294, 394, 149, 312, 58]) reward=1.00 done=false error=null
[STEP] step=2 action=recommend([63, 80, 182, 303, 220]) reward=0.60 done=false error=null
...
[END] success=true steps=10 score=0.540 rewards=1.00,0.60,0.60,0.60,0.40,0.60,0.40,0.00,0.60,0.60
```

---

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t content-rec-env:latest .
```

### Run Container

```bash
# Easy task
docker run -e TASK_NAME=easy content-rec-env:latest

# Medium task
docker run -e TASK_NAME=medium content-rec-env:latest

# Hard task
docker run -e TASK_NAME=hard content-rec-env:latest
```

### Push to Hugging Face Spaces

```bash
# Build and tag for HF registry
docker tag content-rec-env:latest registry.hf.space/your-username/content-rec:latest

# Login to HF
huggingface-cli login

# Push
docker push registry.hf.space/your-username/content-rec:latest
```

---

## 📋 File Structure

```
content-rec-openenv/
├── content_rec_env.py          # Core environment implementation
├── inference.py                # Baseline agent + evaluation harness
├── openenv.yaml                # OpenEnv specification
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── example_scores.json         # Baseline results
```

---

## 🎓 Example: Custom Agent

To build your own agent, implement the recommendation strategy:

```python
from content_rec_env import ContentRecEnvSync, RecommendationAction, TaskDifficulty
import numpy as np

class SmartRecommender:
    def recommend(self, observation):
        """
        observation.user_state: UserState with preferences, history, satisfaction
        observation.available_items: List of 500 ContentItems
        
        Returns: List of 5 item IDs
        """
        user_state = observation.user_state
        items = observation.available_items
        
        # Strategy: weighted score combining user preference + freshness + diversity
        scores = []
        for item in items:
            pref = user_state.genre_preferences.get(item.genre, 0.5)
            freshness = 1.0 - min(item.freshness / 90.0, 1.0)
            score = 0.6 * pref + 0.4 * freshness
            scores.append((item.item_id, score))
        
        # Sort and pick top-5
        scores.sort(key=lambda x: x[1], reverse=True)
        return [item_id for item_id, _ in scores[:5]]

# Use it
env = ContentRecEnvSync(task_difficulty=TaskDifficulty.MEDIUM)
recommender = SmartRecommender()

result = env.reset()
for step in range(10):
    action = RecommendationAction(
        recommended_items=recommender.recommend(result.observation)
    )
    result = env.step(action)
    print(f"Step {step+1}: reward={result.reward:.3f}")
    if result.done:
        break

env.close()
```

---

## 🔧 Customization

### Change Number of Items

Edit `content_rec_env.py`:
```python
# In ContentCatalog._generate_items()
for item_id in range(1000):  # Change from 500 to 1000
    ...
```

### Add New Reward Function

```python
class CustomReward(RewardFunction):
    def compute(self, user_state, recommended_items, clicked_items, engagement, catalog):
        # Your logic
        reward = ...
        return reward, {"custom_metric": value}

# Use it
class ContentRecEnv:
    def __init__(self, reward_fn: RewardFunction):
        self.reward_fn = reward_fn
```

### Adjust Episode Length

Edit `ContentRecEnv.__init__()`:
```python
self.max_steps = 20  # Instead of 10
```

---

## 📊 Metrics

The environment tracks:

| Metric | Range | Description |
|--------|-------|-------------|
| `score` | [0.0, 1.0] | Normalized average reward |
| `ctr` | [0.0, 1.0] | Click-through rate |
| `engagement` | [0.0, 1.0] | User engagement from clicks |
| `diversity` | [0.0, 1.0] | Genre diversity of recommendations |
| `satisfaction` | [0.0, 1.0] | User satisfaction (hard task only) |
| `churn_risk` | [0.0, 1.0] | Estimated churn probability (hard task only) |

---

## 🤝 Contributing

Have ideas? Submit issues and PRs at the GitHub repo!

**Areas for improvement:**
- Multi-user sessions (agents recommend for multiple users simultaneously)
- Cold-start user / new content scenarios
- A/B testing framework (compare two agents on same user trajectories)
- Batch recommendation (recommend top-K instead of 5)

---

## 📄 License

MIT License — see LICENSE file

---

## 🙋 FAQ

**Q: Why only 5 recommendations?**
A: Matches real-world UX (Netflix rows, YouTube homepage, etc.). Can modify in code.

**Q: Are users deterministic?**
A: No — clicks are stochastic. Set `seed` parameter for reproducibility.

**Q: Can I use this with reinforcement learning libraries (Ray, Stable Baselines3)?**
A: Yes! Wrap `ContentRecEnv` in an adapter (e.g., `gym.Wrapper`).

**Q: What's the minimum baseline score to publish?**
A: ~0.3 on easy, ~0.2 on medium, ~0.25 on hard (beat random + simplistic heuristics).

---

**Last Updated:** April 2025  
**Maintainer:** OpenEnv Community  
**Version:** 1.0.0
