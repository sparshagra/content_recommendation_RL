# Quick Start: Content Recommendation Environment

Get up and running in under 5 minutes.

## Step 1: Install (1 min)

```bash
# Extract files from outputs
cd content-rec-env

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Test Locally (1 min)

```bash
# Test the environment
python content_rec_env.py
# Output: "Test passed!"

# Run baseline agent
python inference.py
# Output: [START] ... [STEP] ... [END]

# Get baseline scores for all tasks
for task in easy medium hard; do
    echo "=== Task: $task ==="
    TASK_NAME=$task python inference.py | grep "^\\[END\\]"
done
```

**Expected Output:**
```
[END] success=true steps=10 score=0.540 rewards=1.00,0.60,...
[END] success=true steps=10 score=0.360 rewards=0.82,0.42,...
[END] success=true steps=10 score=0.506 rewards=1.00,0.70,...
```

## Step 3: Integrate Your Agent (2 min)

Create `my_agent.py`:

```python
from content_rec_env import ContentRecEnvSync, RecommendationAction, TaskDifficulty
import random

# Initialize environment
env = ContentRecEnvSync(task_difficulty=TaskDifficulty.EASY)
result = env.reset()

# Run episode
for step in range(10):
    # Your agent: pick 5 items
    # (here: random, but your agent can be smarter)
    recommendations = random.sample(range(500), 5)
    
    # Execute
    action = RecommendationAction(recommended_items=recommendations)
    result = env.step(action)
    
    # Log
    print(f"Step {step+1}: reward={result.reward:.3f}, clicks={len(result.observation.last_clicks or [])}")
    
    if result.done:
        break

env.close()
print(f"Episode summary: {env.get_episode_summary()}")
```

Run it:
```bash
python my_agent.py
```

## Step 4: Deploy to HF Spaces (Optional, 5 min)

See `DEPLOYMENT_GUIDE.md` for detailed instructions. Quick version:

```bash
# 1. Create space at https://huggingface.co/new-space
#    (select Docker SDK)

# 2. Add files
git clone https://huggingface.co/spaces/{your-username}/content-recommendation-env
cp *.py *.yaml *.txt Dockerfile ./
cd content-recommendation-env
git add .
git commit -m "Initial commit"
git push

# 3. Space auto-deploys (2-3 minutes)
# Done! View at: https://huggingface.co/spaces/{your-username}/content-recommendation-env
```

---

## 📚 Learn More

| Resource | Link |
|----------|------|
| Full API | See `README.md` |
| File Overview | See `INDEX.md` |
| Deployment | See `DEPLOYMENT_GUIDE.md` |
| Baseline Scores | See `example_scores.json` |

---

## 🎯 Key Concepts in 30 Seconds

**Task 1 (Easy):** Maximize clicks. Reward = CTR.
```python
# Strategy: Recommend popular items + user preferences
scores = [0.5*popularity + 0.5*genre_preference for item in items]
recommend(top_5_by_score)
```

**Task 2 (Medium):** Maximize clicks while diversifying. Reward = CTR - diversity_penalty.
```python
# Strategy: Balance popularity with genre diversity
recommendations = pick_top_from_each_genre()
```

**Task 3 (Hard):** Maximize long-term retention. Reward = CTR + satisfaction - churn_penalty.
```python
# Strategy: Model user state evolution
# Reduce churn by avoiding fatigue + recommending fresh content
```

---

## 🔍 Typical Agent Development Workflow

```
1. Understand task (read reward function)
        ↓
2. Run baseline (score ~0.5)
        ↓
3. Build simple heuristic (score ~0.6-0.7)
        ↓
4. Train RL agent with Stable Baselines3 (score ~0.8+)
        ↓
5. Deploy to HF Spaces
        ↓
6. Share results + code!
```

---

## ❓ FAQ

**Q: How many items in the catalog?**
A: 500 items (modifiable in code)

**Q: How long is each episode?**
A: 10 steps (modifiable)

**Q: Is the environment stochastic?**
A: Yes, but reproducible with seed=42

**Q: Can I use this with gymnasium/gym?**
A: Yes, wrap the environment (see README for example)

**Q: What's the minimum score to publish?**
A: Easy >0.3, Medium >0.2, Hard >0.25 (beat random)

---

## 📞 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Slow inference | Reduce max_steps or num_items |
| Baseline doesn't match docs | Ensure seed=42 and no code changes |
| Docker build fails | Check `requirements.txt` and Python version |

---

**Ready?** Start with Step 1 above and you'll be running your own agent in minutes! 🚀

See `README.md` for complete API documentation and advanced examples.
