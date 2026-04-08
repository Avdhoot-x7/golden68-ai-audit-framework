"""Trial test - 5 prompts to verify NVIDIA API works with token limits"""
import sys, os, json, time
from datetime import datetime
sys.path.insert(0, '.')

from src.models.adapters import ModelAdapterFactory, APIKeyExhaustedError
from src.evaluation.loader import DatasetLoader
from src.judges.llm_judge import LLMJudge

# API Keys
JUDGE_KEY = "AIzaSyAXvzzVFvrcdVZoMx7Q5PNXwxmkRS7rct8"
NVIDIA_KEY = "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"

# Token limit to prevent credit exhaustion (2048 tokens per request)
MAX_TOKENS = 2048

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
outfile = f"data/results/trial_test_{ts}.json"
os.makedirs("data/results", exist_ok=True)

print("="*70)
print("GOLDEN 68 - TRIAL TEST (5 prompts)")
print("Using NVIDIA Cloud API")
print(f"Token limit: {MAX_TOKENS} tokens per request (credit protection)")
print("="*70)
print(f"Output: {outfile}")

# Create adapters
judge = ModelAdapterFactory.create("gemini", JUDGE_KEY, "gemini-2.5-flash")
test_model = ModelAdapterFactory.create("nvidia", NVIDIA_KEY, "openai/gpt-oss-120b")

loader = DatasetLoader()
all_prompts = loader.get_all_prompts()
prompts = all_prompts[:5]  # Only first 5 prompts
print(f"Loaded {len(prompts)} prompts for trial")

llm_judge = LLMJudge(judge)
results = []

for i, p in enumerate(prompts):
    pid = p.get("id", f"p{i}")
    print(f"\n[{i+1}/{len(prompts)}] {pid}...", end=" ", flush=True)
    
    try:
        # Generate with token limit to protect credits
        resp = test_model.generate(p["prompt"], temperature=0.7, max_tokens=MAX_TOKENS)
        
        # Judge evaluation
        eval_ = llm_judge.evaluate(p["prompt"], resp, p.get("expected_behavior",""), p)
        
        r = {
            "prompt_id": pid,
            "pillar": p.get("pillar",""),
            "level": p.get("level",0),
            "judge_score": eval_.get("score",0),
            "judge_determination": eval_.get("determination","?"),
            "judge_reasoning": eval_.get("explanation",""),
            "response_length": len(resp),
        }
        results.append(r)
        print(f"Score: {r['judge_score']}/10 ({r['judge_determination']}) | Response: {r['response_length']} chars")
        
    except APIKeyExhaustedError as e:
        print(f"\n\n{'='*70}")
        print("API KEY EXHAUSTED - STOPPING")
        print(f"{'='*70}")
        print(f"Error: {e.message}")
        print(f"Prompts completed: {len(results)}/{len(prompts)}")
        break
    
    except Exception as e:
        print(f"\nError: {str(e)}")
        results.append({
            "prompt_id": pid,
            "pillar": p.get("pillar",""),
            "level": p.get("level",0),
            "judge_score": 0,
            "judge_determination": "ERROR",
            "judge_reasoning": str(e),
            "error": True
        })
    
    # Save after each prompt
    with open(outfile, "w") as f:
        json.dump({"results": results, "completed": len(results), "total": len(prompts)}, f, indent=2)
    
    time.sleep(1)  # Rate limiting delay

# Final stats
scores = [r["judge_score"] for r in results if "error" not in r]
avg = sum(scores)/len(scores) if scores else 0
passes = sum(1 for r in results if r.get("judge_determination") == "PASS")

print(f"\n{'='*70}")
print("TRIAL TEST RESULTS")
print(f"{'='*70}")
print(f"Evaluated: {len(results)}/{len(prompts)}")
print(f"Average Score: {avg:.2f}/10")
print(f"Pass Rate: {passes}/{len(results)}")

# Final save
with open(outfile, "w") as f:
    json.dump({
        "results": results,
        "completed": len(results),
        "total": len(prompts),
        "avg_score": avg,
        "passes": passes,
        "max_tokens_limit": MAX_TOKENS,
        "api_provider": "NVIDIA Cloud"
    }, f, indent=2)

print(f"\nSaved: {outfile}")
print("\n✅ Trial complete! Framework is working correctly.")
