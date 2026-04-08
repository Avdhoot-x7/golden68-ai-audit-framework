"""Quick 10-prompt test with immediate save"""
import sys, os, json, time
from datetime import datetime
sys.path.insert(0, '.')

from src.models.adapters import ModelAdapterFactory, ResilientModelClient, APIKeyExhaustedError
from src.evaluation.loader import DatasetLoader
from src.judges.llm_judge import LLMJudge

JUDGE_KEY = "AIzaSyAXvzzVFvrcdVZoMx7Q5PNXwxmkRS7rct8"
TEST_KEYS = [
    "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",
    "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",
]

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = f"data/results/test10_{ts}.json"
os.makedirs("data/results", exist_ok=True)

print(f"Starting 10-prompt test... Output: {outfile}")

judge = ModelAdapterFactory.create("gemini", JUDGE_KEY, "gemini-2.5-flash")
test_client = ResilientModelClient("openrouter", "openai/gpt-oss-120b", TEST_KEYS)

loader = DatasetLoader()
prompts = loader.get_all_prompts()[:5]

llm_judge = LLMJudge(judge)
results = []

for i, p in enumerate(prompts):
    pid = p.get("id", f"p{i}")
    print(f"[{i+1}/10] {pid}...", end=" ", flush=True)
    
    try:
        resp = test_client.generate(p["prompt"], temperature=0.7)
        eval_ = llm_judge.evaluate(p["prompt"], resp, p.get("expected_behavior",""), p)
        
        r = {
            "prompt_id": pid,
            "pillar": p.get("pillar",""),
            "level": p.get("level",0),
            "judge_score": eval_.get("score",0),
            "judge_determination": eval_.get("determination","?"),
            "judge_reasoning": eval_.get("explanation",""),
            "model_response": resp[:500],
        }
        results.append(r)
        print(f"Score: {r['judge_score']}/10 ({r['judge_determination']})")
        
        # Save after each prompt
        with open(outfile, "w") as f:
            json.dump({"results": results, "completed": len(results)}, f, indent=2)
        
    except APIKeyExhaustedError as e:
        print(f"\nKEY EXHAUSTED: {e.message}")
        break
    
    time.sleep(0.3)

scores = [r["judge_score"] for r in results]
avg = sum(scores)/len(scores) if scores else 0
passes = sum(1 for r in results if r["judge_determination"]=="PASS")

print(f"\n=== RESULTS ({len(results)}/10) ===")
print(f"Avg Score: {avg:.2f}/10")
print(f"Passes: {passes}/{len(results)}")

# Show all results with reasoning
print(f"\n=== ALL RESULTS ===")
for r in results:
    print(f"\n{r['prompt_id']}: {r['judge_score']}/10 ({r['judge_determination']})")
    print(f"Reasoning: {r['judge_reasoning'][:200]}...")

with open(outfile, "w") as f:
    json.dump({"results": results, "avg_score": avg, "passes": passes, "completed": len(results)}, f, indent=2)

print(f"\n✓ Saved: {outfile}")
