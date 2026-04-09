---
title: Content Recommendation OpenEnv
emoji: 🎯
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - reinforcement-learning
  - openenv
  - recommendation
  - fastapi
---

# Content Recommendation OpenEnv

> **Meta × PyTorch OpenEnv Hackathon 2026** — A production-quality RL environment for content recommendation.

An agent learns to recommend content (articles, videos, songs) to simulated users over multi-step sessions. Reward signals cover click-through rate, content diversity, and long-term user retention — three real metrics used by recommendation teams at scale.

[![HF Space](https://img.shields.io/badge/🤗%20Space-dikshi2025%2Fcontent--rec-blue)](https://huggingface.co/spaces/dikshi2025/content-rec)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://huggingface.co/spaces/dikshi2025/content-rec/blob/main/openenv.yaml)

---

## Environment Overview

| Property | Value |
|---|---|
| **Action space** | 5 unique item IDs from catalog (0–499) |
| **Observation** | User state + available items + last action/clicks |
| **Reward** | Continuous [0.0, 1.0] — task-dependent |
| **Episode length** | 10 steps |
| **Tasks** | 3 (easy / medium / hard) |
| **Graders** | 3 deterministic graders |

---

## Tasks

| Task | Difficulty | Reward Signal | Goal |
|---|---|---|---|
| `easy_task` | Easy | CTR only | Maximise click-through rate |
| `medium_task` | Medium | CTR − diversity penalty | Balance clicks with genre diversity |
| `hard_task` | Hard | CTR + satisfaction − churn_risk + diversity | Prevent user churn while maximising long-term value |

### Reward Formulas

**Easy** — pure CTR:
```
reward = clicks / 5
```

**Medium** — CTR with diversity requirement:
```
reward = CTR − 0.3 × (1 − genre_diversity)
```

**Hard** — lifetime value:
```
reward = CTR + 0.4 × satisfaction − 0.5 × churn_risk + 0.2 × diversity
```

---

## User Dynamics

Each episode simulates a unique user with:
- **Genre preferences** — 5 weighted scores (pop, tech, sports, entertainment, news)
- **Interaction history** — last 5 clicked items
- **Satisfaction** — moving average of engagement (decays without good recommendations)
- **Fatigue** — rises when same genre is recommended repeatedly
- **Churn risk** — inversely related to satisfaction; spikes with poor sessions

---

## API Reference

The server exposes a standard OpenEnv HTTP API:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `GET` | `/metadata` | Environment metadata |
| `GET` | `/schema` | Action / observation / state JSON schemas |
| `GET` | `/tasks` | List all tasks with grader status |
| `POST` | `/reset` | Start a new episode → returns observation + `session_id` |
| `POST` | `/step` | Execute action → returns observation + reward |
| `GET` | `/state` | Current user state (no side effects) |
| `GET` | `/catalog` | Full 500-item catalog |
| `GET` | `/graders` | List all graders |
| `GET` | `/grader` | Grade a completed session (pass `session_id`) |
| `POST` | `/grade` | Grade actions for a specific task |
| `GET` | `/grade/{task}` | Grade via GET (self-play baseline) |
| `POST` | `/grade/all` | Grade all three tasks at once |
| `GET` | `/docs` | Swagger / OpenAPI UI |

---

## Quick Start

### 1. Reset and step (HTTP)

```python
import requests

BASE = "https://dikshi2025-content-rec.hf.space"

# Start episode
resp = requests.post(f"{BASE}/reset", json={"task": "easy"})
session_id = resp.json()["session_id"]
obs = resp.json()["observation"]

# Step
resp = requests.post(f"{BASE}/step", json={
    "recommended_items": [0, 1, 2, 3, 4],
    "session_id": session_id
})
print(resp.json()["reward"])

# Grade
resp = requests.get(f"{BASE}/grader", params={"session_id": session_id})
print(resp.json()["score"])
```

### 2. Run inference

```bash
export ENV_API_URL=https://dikshi2025-content-rec.hf.space
export MODEL_NAME=HuggingFaceH4/zephyr-7b-beta
export HF_TOKEN=<your_token>
export TASK_NAME=all

python inference.py
```

### 3. Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --port 7860
```

---

## Project Structure

```
├── app.py               # FastAPI app factory (entry point)
├── content_rec_env.py   # Core RL environment (MDP logic)
├── graders.py           # Deterministic graders for all 3 tasks
├── inference.py         # LLM agent runner (OpenEnv spec compliant)
├── env_state.py         # Shared state and serialisation helpers
├── models.py            # Pydantic models (OpenEnv spec typed)
├── routes/
│   ├── meta_routes.py      # /health, /metadata, /schema, /tasks
│   ├── env_routes.py       # /reset, /step, /state, /catalog
│   └── grader_routes.py    # /grader, /graders, /grade, /grade/all
├── server/              # Server entry point shim
├── tasks/               # Per-task YAML configs (grader refs)
│   ├── easy.yaml
│   ├── medium.yaml
│   └── hard.yaml
├── openenv.yaml         # OpenEnv spec declaration
├── Dockerfile           # Container build
├── requirements.txt     # Dependencies
└── docs/                # Extended documentation
```

---

## Graders

All graders are **deterministic and reproducible** — same seed → same score.

```python
from graders import easy_task_grader, medium_task_grader, hard_task_grader

# Grade with agent actions
actions = [[0,1,2,3,4]] * 10  # list of 10 steps, each 5 item IDs
result = easy_task_grader(actions=actions)
print(result["score"])   # float in [0.0, 1.0]
print(result["success"]) # bool

# Or grade via API
GET /grade/easy_task  # runs self-play baseline
```

---

## Baseline Scores (heuristic agent)

| Task | Heuristic Score | Success Threshold |
|---|---|---|
| easy_task | ~0.54 | 0.25 |
| medium_task | ~0.36 | 0.15 |
| hard_task | ~0.51 | 0.20 |

Scores above the threshold indicate the agent outperforms a popularity-based heuristic.
