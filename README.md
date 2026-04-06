---
title: Content Recommendation OpenEnv
emoji: 🎯
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - reinforcement-learning
  - openenv
  - content-recommendation
  - rl-environment
  - meta-pytorch-hackathon
  - recommendation-system
---

# 🎯 Content Recommendation — OpenEnv RL Environment

> **Meta × PyTorch OpenEnv Hackathon** submission by [@dikshi2025](https://huggingface.co/dikshi2025)

A **production-grade reinforcement learning environment** where an AI agent learns to recommend content (articles, videos, songs) to simulated users — optimising for engagement, diversity, and long-term retention.

Real-world relevance: **Netflix, YouTube, TikTok, Spotify** all solve exactly this problem.

---

## 🚀 Try It Live

The environment exposes a REST API. No setup required — just call the endpoints below.

### 1. Health Check
```bash
curl https://dikshi2025-content-rec.hf.space/health
```
```json
{ "status": "ok", "environment": "content-recommendation", "version": "1.0.0" }
```

### 2. Start a New Episode
```bash
curl -X POST https://dikshi2025-content-rec.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy", "seed": 42}'
```

### 3. Take a Step (Recommend 5 Items)
```bash
curl -X POST https://dikshi2025-content-rec.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"recommended_items": [42, 101, 250, 388, 17]}'
```

### 4. Get Current User State
```bash
curl https://dikshi2025-content-rec.hf.space/state
```

📖 Full interactive API docs: [`/docs`](https://dikshi2025-content-rec.hf.space/docs)

---

## 🧠 What This Environment Models

A streaming platform session where:

| Component | Description |
|---|---|
| **Users** | 1,000 simulated users with genre preferences, fatigue, and churn risk |
| **Content** | 500 items across 5 genres (pop, tech, sports, entertainment, news) |
| **Agent** | Recommends 5 items per step, receives reward signal |
| **Session** | 10 steps per episode, user state evolves over time |

---

## 📊 Three Difficulty Tasks

### 🟢 Task 1 — Easy: CTR Optimization
Pure click-through rate. Greedy strategies work well.
```
reward = (items clicked) / 5          → range [0.0, 1.0]
baseline score: ~0.54
```

### 🟡 Task 2 — Medium: Engagement + Diversity
Must balance clicks *and* genre diversity to avoid filter bubbles.
```
reward = CTR − 0.3 × (1 − genre_diversity)   → range [0.0, 1.0]
baseline score: ~0.36
```

### 🔴 Task 3 — Hard: Long-term User Lifetime Value
Models temporal dynamics: satisfaction, fatigue, and churn risk compound over steps.
```
reward = CTR + 0.4×satisfaction − 0.5×churn_risk + 0.2×diversity
baseline score: ~0.51
```

> All rewards are in **[0.0, 1.0]** with partial progress signals at every step.

---

## 🔄 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check — returns HTTP 200 |
| `POST` | `/reset` | Start new episode, choose task + seed |
| `POST` | `/step` | Submit 5 item IDs, get reward + next state |
| `GET` | `/state` | Current user state (no step taken) |
| `GET` | `/episode_summary` | Cumulative episode metrics |
| `GET` | `/docs` | Interactive Swagger UI |

### Action Space
```json
{ "recommended_items": [42, 101, 250, 388, 17] }
```
- Exactly **5 unique item IDs**
- IDs in range **[0, 499]**

### Observation Space
```json
{
  "user_state": {
    "user_id": 731,
    "genre_preferences": { "pop": 0.82, "tech": 0.21, "sports": 0.74, "entertainment": 0.55, "news": 0.13 },
    "satisfaction": 0.62,
    "fatigue": 0.15,
    "churn_risk": 0.28,
    "interaction_history": [42, 101, 17],
    "session_step": 3
  },
  "last_clicks": [42, 17],
  "last_reward": 0.60,
  "done": false
}
```

---

## 🤖 Baseline Agent

The bundled `inference.py` uses **LLM-guided recommendations** (via `HuggingFaceH4/zephyr-7b-beta`) with an automatic heuristic fallback:

```
score(item) = 0.5 × popularity
            + 0.3 × genre_preference_match
            + 0.2 × freshness_bonus
            − genre_repeat_penalty
            − recent_recommendation_penalty
```

**Run locally:**
```bash
# Set env vars
export API_BASE_URL=https://api-inference.huggingface.co/v1
export HF_TOKEN=your_hf_token
export MODEL_NAME=HuggingFaceH4/zephyr-7b-beta

# Easy task
TASK_NAME=easy python inference.py

# Medium task
TASK_NAME=medium python inference.py

# Hard task
TASK_NAME=hard python inference.py
```

Expected output:
```
[START] task=easy env=content-rec model=HuggingFaceH4/zephyr-7b-beta
[STEP] step=1 action=recommend([294, 394, 149, 312, 58]) reward=1.00 done=false error=null
[STEP] step=2 action=recommend([63, 80, 182, 303, 220]) reward=0.60 done=false error=null
...
[END] success=true steps=10 score=0.540 rewards=1.00,0.60,0.60,0.60,0.40,0.60,0.40,0.00,0.60,0.60
```

---

## 📁 File Structure

```
├── content_rec_env.py     # Core RL environment (reset/step/state + reward functions)
├── inference.py           # Baseline agent (LLM + heuristic fallback)
├── app.py                 # FastAPI server — exposes environment over HTTP
├── openenv.yaml           # OpenEnv specification (tasks, action/obs space)
├── Dockerfile             # Docker build for HF Spaces
├── requirements.txt       # Python dependencies
├── example_scores.json    # Reproducible baseline scores
└── README.md              # Full technical documentation
```

---

## 🐳 Docker

```bash
# Build
docker build -t content-rec-env .

# Run server (same as HF Spaces)
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e TASK_NAME=easy \
  content-rec-env

# Verify
curl localhost:7860/health
```

---

## 📈 User Behaviour Model

**Click probability per item:**
```
base_ctr     = 0.30 + 0.30 × genre_preference + 0.20 × item_popularity
novelty      = 0.10 × (1 − freshness/90)
fatigue_pen  = user_fatigue × 0.15

P(click)     = max(0, base_ctr + novelty − fatigue_pen)
```

**State evolution after each step:**
```
satisfaction = 0.7 × old_satisfaction + 0.3 × engagement
churn_risk   = max(0, 0.15 + 0.7 × (1 − satisfaction))
fatigue      = min(1, fatigue + 0.15) − 0.05   # builds up, slowly decays
```

---

## 🏆 Beat the Baseline

| Task | Baseline | Target to beat |
|---|---|---|
| Easy (CTR) | 0.54 | > 0.65 |
| Medium (Diversity) | 0.36 | > 0.45 |
| Hard (LTV + Churn) | 0.51 | > 0.60 |

**Strategies:**
- **Easy** → Pure popularity + preference matching
- **Medium** → Force genre rotation every 2-3 steps
- **Hard** → Track churn risk trend; lower fatigue before it spikes

---

## 📄 License

MIT — free to use, fork, and build on.

---

*Built for Meta × PyTorch OpenEnv Hackathon 2026 · Environment v1.0.0*
