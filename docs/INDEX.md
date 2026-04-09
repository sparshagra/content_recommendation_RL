# Content Recommendation Environment - Complete System

## 📦 What You Have

A **production-grade OpenEnv environment** for training AI agents on content recommendation. Three difficulty levels test different skills: CTR optimization → diversity management → long-term churn prevention.

---

## 📋 File Structure & Purpose

### Core Environment

| File | Size | Purpose |
|------|------|---------|
| **content_rec_env.py** | 18 KB | Main environment implementation. Contains: `ContentRecEnv` class (async), user/content simulators, 3 reward functions, all OpenEnv-compliant types. Entry point for integration. |
| **openenv.yaml** | 5.5 KB | Environment specification. Defines observation space, action space, task descriptions, baseline scores, metadata. Used by OpenEnv frameworks for registration. |

### Inference & Evaluation

| File | Size | Purpose |
|------|------|---------|
| **inference.py** | 7.7 KB | Baseline agent using heuristic recommendations. Implements OpenEnv stdout format (`[START]`, `[STEP]`, `[END]` lines). Reproduces baseline scores: Easy 0.54, Medium 0.36, Hard 0.51. |
| **example_scores.json** | 3.3 KB | Baseline results and performance metrics. Shows expected score ranges for all three tasks. |
| **eval_baseline.sh** | 1 KB | Shell script to run evaluation across all three tasks and collect statistics. |

### Deployment

| File | Size | Purpose |
|------|------|---------|
| **Dockerfile** | 604 B | Docker configuration for HuggingFace Spaces. Installs deps, copies code, sets environment variables. |
| **requirements.txt** | 51 B | Python package dependencies (numpy, pyyaml). Minimal footprint. |

### Documentation

| File | Size | Purpose |
|------|------|---------|
| **README.md** | 15 KB | Complete guide covering: quick start, API reference, reward functions, user simulation, baseline agent, Docker, custom agents, FAQ. |
| **DEPLOYMENT_GUIDE.md** | New | Step-by-step instructions for deploying to HuggingFace Spaces. Includes troubleshooting and advanced topics. |
| **INDEX.md** | This file | Overview of what was created and how files fit together. |

---

## 🎯 Three Difficulty Tasks

### Task 1: Easy - Pure CTR Optimization
```
Reward = (clicks / 5)
Baseline: 0.540
Success threshold: 0.25
```
**Challenge:** Maximize immediate clicks. No temporal complexity.
**Strategy:** Recommend popular items matching user's top genre preferences.

### Task 2: Medium - CTR + Diversity
```
Reward = CTR - 0.3 * (1 - genre_diversity)
Baseline: 0.360
Success threshold: 0.15
```
**Challenge:** Balance clicks with content diversity (avoid filter bubbles).
**Strategy:** Mix genres even if less popular. Track recent recommendations to diversify.

### Task 3: Hard - Long-term Value + Churn
```
Reward = CTR + 0.4*satisfaction - 0.5*churn_penalty + 0.2*diversity
Baseline: 0.506
Success threshold: 0.20
```
**Challenge:** Maximize long-term user retention (prevent churn) while maintaining engagement.
**Strategy:** Model how recommendations affect future user state. Balance novelty vs. safety.

---

## 🏃 Getting Started (5 Minutes)

### 1. Test Locally

```bash
# Install
pip install -r requirements.txt

# Run environment test
python content_rec_env.py

# Run baseline inference
TASK_NAME=easy python inference.py
TASK_NAME=medium python inference.py
TASK_NAME=hard python inference.py
```

Expected output: `[START]`, `[STEP]` lines, `[END]` with score.

### 2. Deploy to HuggingFace Spaces

```bash
# Follow DEPLOYMENT_GUIDE.md
# (1) Create space on huggingface.co
# (2) Git push files
# (3) HF auto-builds Docker image
# (4) Space runs inference.py and logs results
```

### 3. Integrate Your Agent

```python
from content_rec_env import ContentRecEnvSync, RecommendationAction, TaskDifficulty

env = ContentRecEnvSync(task_difficulty=TaskDifficulty.MEDIUM)
result = env.reset()

for step in range(10):
    # YOUR AGENT LOGIC HERE
    recommendations = your_agent(result.observation)
    
    action = RecommendationAction(recommended_items=recommendations)
    result = env.step(action)
    print(f"Step {step+1}: reward={result.reward:.3f}")
    if result.done:
        break

env.close()
```

---

## 🔌 API Overview

### Environment

```python
# Async (recommended for production)
from content_rec_env import ContentRecEnv, TaskDifficulty
env = ContentRecEnv(task_difficulty=TaskDifficulty.EASY)
result = await env.reset()
result = await env.step(action)
await env.close()

# Sync (easier for scripts)
from content_rec_env import ContentRecEnvSync
env = ContentRecEnvSync(task_difficulty=TaskDifficulty.EASY)
result = env.reset()
result = env.step(action)
env.close()
```

### Types

```python
# Action: Recommend 5 items
action = RecommendationAction(recommended_items=[0, 1, 2, 3, 4])

# Observation: User state + catalog
observation.user_state.genre_preferences  # Dict[str, float]
observation.user_state.satisfaction       # float [0-1]
observation.user_state.churn_risk         # float [0-1]
observation.available_items               # List[ContentItem] (500 items)

# Step result
result.observation  # ContentRecObservation
result.reward       # float [0-1]
result.done         # bool
result.info         # Dict with task-specific metrics
```

---

## 📊 Metrics & Evaluation

| Metric | Task | Description |
|--------|------|-------------|
| score | All | Normalized average reward [0-1] |
| ctr | All | Click-through rate |
| engagement | All | User engagement from clicks |
| diversity | Medium, Hard | Genre diversity of recommendations |
| satisfaction | Hard | User satisfaction trajectory |
| churn_risk | Hard | Estimated churn probability |

**Evaluation Protocol:**
1. Run 5+ episodes per task
2. Average scores
3. Compare vs. baseline
4. Publish results with reproducible seed=42

---

## 🔧 Customization Examples

### Change Task Difficulty

```python
env = ContentRecEnvSync(
    task_difficulty=TaskDifficulty.HARD,
    seed=42
)
```

### Use Custom Reward Function

```python
from content_rec_env import RewardFunction

class MyReward(RewardFunction):
    def compute(self, user_state, items, clicks, engagement, catalog):
        reward = len(clicks) / 5  # Simple CTR
        return reward, {}

env.reward_fn = MyReward()
```

### Modify Episode Length

```python
env = ContentRecEnvSync(...)
env.env.max_steps = 20  # Instead of 10
```

---

## 🚀 Advanced: RL Training

To train an RL agent (e.g., PPO, DQN), wrap the environment:

```python
import gym
from stable_baselines3 import PPO

# Wrap for gym
class GymWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.action_space = gym.spaces.MultiDiscrete([500]*5)
        # Define observation_space...
    
    def reset(self):
        result = self.env.reset()
        return self._obs_to_array(result.observation)
    
    def step(self, action):
        act = RecommendationAction(list(action))
        result = self.env.step(act)
        return (
            self._obs_to_array(result.observation),
            result.reward,
            result.done,
            {}
        )

# Train
env = GymWrapper(ContentRecEnvSync())
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

---

## 📈 Baseline Performance

| Task | Model | Score | Strategy |
|------|-------|-------|----------|
| Easy | heuristic | 0.540 | Popularity + preference matching |
| Medium | heuristic | 0.360 | Add diversity penalty (loses CTR) |
| Hard | heuristic | 0.506 | Multi-objective: engagement + churn |

**To Beat Baselines:**
- Easy: >0.60 (improve preference matching)
- Medium: >0.45 (explicit diversity planning)
- Hard: >0.55 (predict churn dynamics)

---

## ✅ Checklist: What's Included

- [x] OpenEnv-compliant environment (`step()`, `reset()`, `state()`)
- [x] Typed models (dataclasses with validation)
- [x] 3 difficulty tasks with graders
- [x] Meaningful reward functions with partial progress signals
- [x] Baseline inference script with reproducible scores
- [x] Dockerfile for HF Spaces deployment
- [x] `openenv.yaml` specification
- [x] Complete README with examples
- [x] Evaluation script (`eval_baseline.sh`)
- [x] JSON baseline results
- [x] Deployment guide

---

## 🎓 Next Steps

1. **Test locally:** `python inference.py` to verify baseline
2. **Deploy:** Follow `DEPLOYMENT_GUIDE.md` for HF Spaces
3. **Customize:** Modify `inference.py` with your agent
4. **Train:** Integrate with Stable Baselines3, Ray, or similar RL library
5. **Publish:** Share your space and results!

---

## 📞 Support

- **Environment issues:** Check README.md FAQ section
- **Deployment help:** See DEPLOYMENT_GUIDE.md
- **RL integration:** Look at example agent code in README.md
- **Contribute:** Fork the repo and submit improvements!

---

## 📄 License

MIT License — free for commercial/academic use

---

**Created:** April 2025  
**Version:** 1.0.0  
**Framework:** OpenEnv  
**Status:** Production-ready ✅
