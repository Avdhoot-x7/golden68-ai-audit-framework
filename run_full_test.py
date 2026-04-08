"""Full 66-prompt test with instant save on each prompt"""
import sys, os, json, time
from datetime import datetime
sys.path.insert(0, '.')

from src.models.adapters import ModelAdapterFactory, ResilientModelClient, APIKeyExhaustedError, AutoRecoveryModelClient
from src.evaluation.loader import DatasetLoader
from src.judges.llm_judge import LLMJudge

JUDGE_KEY = "AIzaSyAXvzzVFvrcdVZoMx7Q5PNXwxmkRS7rct8"
NVIDIA_KEY = "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"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = f"data/results/full_test_{ts}.json"
os.makedirs("data/results", exist_ok=True)

print("="*70)
print("GOLDEN 68 - FULL EVALUATION (66 prompts)")
print("Using NVIDIA Cloud API (rate limited to prevent exhaustion)")
print("="*70)
print(f"Output: {outfile}")

judge = ModelAdapterFactory.create("gemini", JUDGE_KEY, "gemini-2.5-flash")
test_client = AutoRecoveryModelClient("nvidia", "openai/gpt-oss-120b", [NVIDIA_KEY])

loader = DatasetLoader()
prompts = loader.get_all_prompts()
print(f"Loaded {len(prompts)} prompts")

llm_judge = LLMJudge(judge)
results = []

for i, p in enumerate(prompts):
    pid = p.get("id", f"p{i}")
    print(f"\n[{i+1}/{len(prompts)}] {pid}...", end=" ", flush=True)
    
    try:
        resp = test_client.generate(p["prompt"], temperature=0.7, max_tokens=2048)
        eval_ = llm_judge.evaluate(p["prompt"], resp, p.get("expected_behavior",""), p)
        
        r = {
            "prompt_id": pid,
            "pillar": p.get("pillar",""),
            "level": p.get("level",0),
            "judge_score": eval_.get("score",0),
            "judge_determination": eval_.get("determination","?"),
            "judge_reasoning": eval_.get("explanation",""),
        }
        results.append(r)
        print(f"Score: {r['judge_score']}/10 ({r['judge_determination']})")
        
    except APIKeyExhaustedError as e:
        print(f"\n\n{'='*70}")
        print("API KEY EXHAUSTED - STOPPING EVALUATION")
        print(f"{'='*70}")
        print(f"Error: {e.message}")
        print(f"Prompts completed: {len(results)}/{len(prompts)}")
        break
    
    # Save after each prompt
    with open(outfile, "w") as f:
        json.dump({"results": results, "completed": len(results), "total": len(prompts)}, f, indent=2)
    
    time.sleep(0.3)

# Final stats
scores = [r["judge_score"] for r in results]
avg = sum(scores)/len(scores) if scores else 0
passes = sum(1 for r in results if r["judge_determination"]=="PASS")

print(f"\n{'='*70}")
print("FINAL RESULTS")
print(f"{'='*70}")
print(f"Evaluated: {len(results)}/{len(prompts)}")
print(f"Average Score: {avg:.2f}/10")
print(f"Pass Rate: {passes}/{len(results)}")

# Pillar stats
pillars = {}
for r in results:
    p = r["pillar"]
    if p not in pillars:
        pillars[p] = {"scores": [], "passes": 0}
    pillars[p]["scores"].append(r["judge_score"])
    if r["judge_determination"] == "PASS":
        pillars[p]["passes"] += 1

print("\nPILLAR BREAKDOWN:")
for p, data in pillars.items():
    avg_p = sum(data["scores"])/len(data["scores"])
    pr = data["passes"]/len(data["scores"])*100
    print(f"  {p}: {avg_p:.2f}/10 | {pr:.1f}% pass ({data['passes']}/{len(data['scores'])})")

# Final save
with open(outfile, "w") as f:
    json.dump({
        "results": results,
        "completed": len(results),
        "total": len(prompts),
        "avg_score": avg,
        "passes": passes,
        "pillars": pillars
    }, f, indent=2)

print(f"\nSaved: {outfile}")
