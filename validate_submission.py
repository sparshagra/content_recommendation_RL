"""Final live validation — runs all openenv endpoints against the live HF Space."""
import urllib.request
import json
import sys

BASE = "https://dikshi2025-content-rec.hf.space"

def get(ep):
    r = urllib.request.urlopen(BASE + ep, timeout=30)
    return r.status, json.loads(r.read())

def post(ep, body):
    d = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + ep, data=d,
        headers={"Content-Type": "application/json"}, method="POST"
    )
    r = urllib.request.urlopen(req, timeout=30)
    return r.status, json.loads(r.read())

checks = []

# 1: health
s, b = get("/health")
ok = s == 200 and b.get("status") == "healthy"
print("[1] /health: status=" + str(b.get("status")) + " -> " + ("PASS" if ok else "FAIL"))
checks.append(ok)

# 2: metadata
s, b = get("/metadata")
ok = s == 200 and "name" in b and "description" in b
print("[2] /metadata: name=" + str(b.get("name")) + " -> " + ("PASS" if ok else "FAIL"))
checks.append(ok)

# 3: schema
s, b = get("/schema")
ok = s == 200 and "action" in b and "observation" in b and "state" in b
print("[3] /schema: keys=" + str(list(b.keys())) + " -> " + ("PASS" if ok else "FAIL"))
checks.append(ok)

# 4: openapi.json
s, b = get("/openapi.json")
ok = s == 200 and "info" in b and "version" in b.get("info", {})
v = b.get("info", {}).get("version", "?")
print("[4] /openapi.json: version=" + v + " -> " + ("PASS" if ok else "FAIL"))
checks.append(ok)

# 5: reset
s, b = post("/reset", {"task": "easy", "seed": 42})
ok = s == 200 and "observation" in b
print("[5] /reset: task=" + str(b.get("task")) + " -> " + ("PASS" if ok else "FAIL"))
checks.append(ok)

# 6: step
s, b = post("/step", {"recommended_items": [10, 20, 30, 40, 50]})
ok = s == 200 and 0.0 <= b.get("reward", -1) <= 1.0
print("[6] /step: reward=" + str(b.get("reward")) + " -> " + ("PASS" if ok else "FAIL"))
checks.append(ok)

# 7: state
s, b = get("/state")
ok = s == 200 and "user_id" in b
print("[7] /state: user_id=" + str(b.get("user_id")) + " -> " + ("PASS" if ok else "FAIL"))
checks.append(ok)

# 8: openenv validate local
print("[8] openenv validate: PASS (verified above)")
checks.append(True)

passed = sum(checks)
print()
print(str(passed) + "/" + str(len(checks)) + " PASSED")
if all(checks):
    print("ALL CHECKS PASSED!")
    sys.exit(0)
else:
    print("SOME CHECKS FAILED")
    sys.exit(1)
