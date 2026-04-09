# Deployment Guide: HuggingFace Spaces

Deploy your content recommendation environment to HuggingFace Spaces in 5 minutes.

## Prerequisites

- HuggingFace account (create at https://huggingface.co)
- Git installed locally
- Docker installed (for local testing)

## Step 1: Prepare Your Repository

Create a new repository on GitHub or locally:

```bash
mkdir content-rec-env
cd content-rec-env

# Copy all files from /outputs
cp content_rec_env.py inference.py openenv.yaml Dockerfile requirements.txt README.md .

# Initialize git
git init
git add .
git commit -m "Initial commit: content recommendation environment"
```

## Step 2: Create HF Spaces Repo

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name:** `content-recommendation-env` (or your choice)
   - **Owner:** Select your username
   - **License:** MIT
   - **Space SDK:** `Docker`
3. Click "Create Space"

## Step 3: Add Files to HF Repo

Option A: Push via Git

```bash
git remote add huggingface https://huggingface.co/spaces/{your-username}/content-recommendation-env
git push huggingface main
```

Option B: Upload Files in UI

1. Go to your space: https://huggingface.co/spaces/{your-username}/content-recommendation-env
2. Click "Files" tab → "Upload files"
3. Select all files from your local folder
4. Commit and push

## Step 4: Configure Docker

Your `Dockerfile` is already included. HF Spaces will automatically:

1. Detect `Dockerfile`
2. Build the image
3. Deploy the container
4. Make it accessible at: `https://huggingface.co/spaces/{your-username}/content-recommendation-env`

## Step 5: Test the Deployment

Once built, your space will run the inference script. Check:

1. **Space URL:** https://huggingface.co/spaces/{your-username}/content-recommendation-env
2. **Build Status:** Shows "Building..." → "Running" (takes ~2-3 minutes)
3. **Logs:** Click "View Logs" to see inference.py output

Expected output:
```
[START] task=easy env=content-rec model=heuristic-baseline
[STEP] step=1 action=recommend([...]) reward=1.00 done=false error=null
...
[END] success=true steps=10 score=0.540 rewards=...
```

## Step 6: Run Multiple Tasks

To test all three tasks, modify your `Dockerfile`:

```dockerfile
# Instead of:
CMD ["python", "inference.py"]

# Use:
CMD ["bash", "-c", "for task in easy medium hard; do echo '=== Task: '$task' ==='; TASK_NAME=$task python inference.py; done"]
```

Or create a separate test script.

## Troubleshooting

### Build fails with "dependency error"

**Solution:** Update `requirements.txt` with all dependencies:
```bash
pip freeze > requirements.txt
```

### Space shows "Building" indefinitely

**Solution:** Check build logs for errors. Common issues:
- Missing `requirements.txt`
- Invalid Python syntax in environment files
- Package version conflicts

### Inference crashes with "ModuleNotFoundError"

**Solution:** Ensure all imports are in `requirements.txt`:
```bash
# Add to requirements.txt if missing:
numpy>=1.21.0
pyyaml>=5.4.0
```

### Output not appearing in logs

**Solution:** Add `flush=True` to print statements (already done in inference.py):
```python
print("[END] ...", flush=True)
```

## Advanced: Custom Gradio UI

Optional: Add a web interface by creating `app.py`:

```python
import gradio as gr
import asyncio
from content_rec_env import ContentRecEnvSync, RecommendationAction, TaskDifficulty

def run_recommendation(task="easy"):
    env = ContentRecEnvSync(task_difficulty=TaskDifficulty[task.upper()])
    result = env.reset()
    
    # Run 3 random steps
    rewards = []
    for _ in range(3):
        action = RecommendationAction([0, 1, 2, 3, 4])
        result = env.step(action)
        rewards.append(f"{result.reward:.3f}")
    
    env.close()
    return f"Task: {task}\nRewards: {rewards}\nScore: {sum(float(r) for r in rewards) / len(rewards):.3f}"

demo = gr.Interface(
    fn=run_recommendation,
    inputs=gr.Dropdown(["easy", "medium", "hard"]),
    outputs="text",
)

if __name__ == "__main__":
    demo.launch()
```

Then update `Dockerfile`:
```dockerfile
CMD ["python", "app.py"]
```

## Scaling & Production

For production deployments:

1. **Resource Limits:** HF Spaces offer free tier with limited compute. For higher demand:
   - Upgrade to "Pro" spaces (GPU/CPU options)
   - Use HF Inference API instead of running inference in-space

2. **Monitoring:** Check space settings for:
   - Build status
   - Uptime metrics
   - Resource usage

3. **Updates:** To update code:
   ```bash
   git add .
   git commit -m "Update environment logic"
   git push huggingface main
   # Space auto-rebuilds
   ```

## Next Steps

1. **Benchmark:** Run evaluation against your agent:
   ```bash
   bash eval_baseline.sh > results.txt
   ```

2. **Improve:** Modify `inference.py` to use RL algorithms (PPO, DQN, etc.)

3. **Share:** Post your space link to the OpenEnv community!

---

**Need Help?**
- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- OpenEnv Issues: https://github.com/openenv/openenv
- Environment README: See README.md for full API reference
