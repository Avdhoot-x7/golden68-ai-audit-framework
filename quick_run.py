"""Quick run evaluation with output"""
import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.models.adapters import ModelAdapterFactory, ResilientModelClient, APIKeyExhaustedError
from src.evaluation.loader import DatasetLoader
from src.judges.llm_judge import LLMJudge

# Keys
JUDGE_KEY = "AIzaSyAXvzzVFvrcdVZoMx7Q5PNXwxmkRS7rct8"
TEST_KEYS = [
    "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",
    "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",
]

print("="*60)
print("GOLDEN 68 - Evaluation")
print("="*60)

# Init
print("\n[1] Initializing...")
judge = ModelAdapterFactory.create("gemini", JUDGE_KEY, "gemini-2.5-flash")
test_client = ResilientModelClient("openrouter", "openai/gpt-oss-120b", TEST_KEYS)
print("✓ Adapters ready")

# Load
print("\n[2] Loading prompts...")
loader = DatasetLoader()
prompts = loader.get_all_prompts()
print(f"✓ {len(prompts)} prompts")

# Evaluate
print("\n[3] Evaluating...")
llm_judge = LLMJudge(judge)
results = []

for i, p in enumerate(prompts):
    pid = p.get("id", f"p{i}")
    print(f"[{i+1}/{len(prompts)}] {pid}...", end=" ", flush=True)
    
    try:
        resp = test_client.generate(p["prompt"], temperature=0.7)
        eval_ = llm_judge.evaluate(p["prompt"], resp, p.get("expected_behavior",""), p)
        
        results.append({
            "prompt_id": pid,
            "pillar": p.get("pillar",""),
            "level": p.get("level",0),
            "prompt": p["prompt"],
            "model_response": resp,
            "judge_score": eval_.get("score",0),
            "judge_determination": eval_.get("determination","?"),
            "judge_reasoning": eval_.get("explanation",""),
        })
        print(f"Score: {eval_.get('score',0)}/10 ({eval_.get('determination','?')})")
        
    except APIKeyExhaustedError as e:
        print(f"\n🔴 KEY EXHAUSTED: {e}")
        print(f"Completed: {len(results)}/{len(prompts)}")
        break
    
    time.sleep(0.3)

# Stats
scores = [r["judge_score"] for r in results]
avg = sum(scores)/len(scores) if scores else 0
passes = sum(1 for r in results if r["judge_determination"]=="PASS")

print("\n"+"="*60)
print("RESULTS")
print("="*60)
print(f"Evaluated: {len(results)}/{len(prompts)}")
print(f"Score: {avg:.2f}/10")
print(f"Passes: {passes}/{len(results)}")

# Save
os.makedirs("data/results", exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

report = {
    "timestamp": ts,
    "judge_model": "gemini-2.5-flash",
    "test_model": "gpt-oss-120b",
    "total": len(prompts),
    "evaluated": len(results),
    "overall_score": round(avg,2),
    "pass_rate": round(passes/len(results)*100,1) if results else 0,
    "passes": passes,
    "results": results
}

with open(f"data/results/eval_{ts}.json","w") as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Saved: data/results/eval_{ts}.json")
