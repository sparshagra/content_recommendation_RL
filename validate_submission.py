"""Final hackathon validation script — runs against live HF Space."""
import urllib.request, json, sys

BASE = "https://dikshi2025-content-rec.hf.space"

def post(ep, body=None):
    d = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{BASE}{ep}", data=d,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    res = urllib.request.urlopen(req, timeout=30)
    return res.status, json.loads(res.read())

def get(ep):
    res = urllib.request.urlopen(f"{BASE}{ep}", timeout=30)
    return res.status, json.loads(res.read())

results = []

print("\n" + "="*62)
print("  FINAL HACKATHON VALIDATION — LIVE HF SPACE")
print("="*62)

# 1. Health
print("\n[1/7] GET /health")
s, b = get("/health")
ok = s == 200 and b.get("status") == "ok"
print(f"  HTTP {s} | {b}")
results.append(("HTTP /health returns 200", ok))

# 2. Reset easy
print("\n[2/7] POST /reset (easy, typed Pydantic response)")
s, b = post("/reset", {"task": "easy", "seed": 42})
obs = b.get("observation", {})
has_user = "user_state" in obs
has_items = "available_items" in obs
has_reward = "reward" in b and "done" in b
print(f"  HTTP {s} | task={b.get('task')} | done={b.get('done')}")
print(f"  user_state fields: {list(obs.get('user_state', {}).keys())}")
print(f"  available_items count: {len(obs.get('available_items', []))}")
results.append(("reset() returns typed Observation", has_user and has_items and has_reward))

# 3. Step
print("\n[3/7] POST /step (reward in [0,1] + typed RewardModel)")
s, b2 = post("/step", {"recommended_items": [10, 20, 30, 40, 50]})
reward_ok = 0.0 <= b2.get("reward", -1) <= 1.0
info_ok = "ctr" in b2.get("info", {}) and "clicks" in b2.get("info", {})
print(f"  HTTP {s} | reward={b2.get('reward')} | done={b2.get('done')}")
print(f"  info (RewardModel): {b2.get('info')}")
results.append(("step() reward in [0.0, 1.0]", reward_ok))
results.append(("step() returns typed RewardModel", info_ok))

# 4. State
print("\n[4/7] GET /state (typed UserStateModel)")
s, b3 = get("/state")
state_ok = "user_id" in b3 and "satisfaction" in b3 and "churn_risk" in b3
print(f"  HTTP {s} | user_id={b3.get('user_id')} | step={b3.get('session_step')}")
results.append(("state() returns typed UserStateModel", state_ok))

# 5. Medium task
print("\n[5/7] POST /reset  task=medium")
s, bm = post("/reset", {"task": "medium", "seed": 42})
print(f"  HTTP {s} | task={bm.get('task')}")
results.append(("reset(medium) task works", s == 200 and bm.get("task") == "medium"))

# 6. Hard task
print("\n[6/7] POST /reset  task=hard")
s, bh = post("/reset", {"task": "hard", "seed": 42})
print(f"  HTTP {s} | task={bh.get('task')}")
results.append(("reset(hard) task works", s == 200 and bh.get("task") == "hard"))

# 7. Catalog
print("\n[7/7] GET /catalog (500 items)")
s, bc = get("/catalog")
genres = list({x["genre"] for x in bc})
print(f"  HTTP {s} | items={len(bc)} | genres={sorted(genres)}")
results.append(("catalog returns 500 items", s == 200 and len(bc) == 500))

# Summary
passed = sum(1 for _, ok in results if ok)
print("\n" + "="*62)
print("  RESULTS")
print("="*62)
for name, ok in results:
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}]  {name}")

print()
print(f"  {passed}/{len(results)} checks passed")
status = "ALL LIVE API TESTS PASSED" if passed == len(results) else "SOME CHECKS FAILED"
print(f"  {status}")
print("="*62)
sys.exit(0 if passed == len(results) else 1)
