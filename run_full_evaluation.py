"""
Run Full Evaluation Script for Golden 68 Framework
With resilient API key handling and fallback support
"""
import sys
import os
import json
import time
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))

from src.models.adapters import (
    ModelAdapterFactory, 
    ResilientModelClient, 
    APIKeyExhaustedError
)
from src.evaluation.loader import DatasetLoader
from src.judges.llm_judge import LLMJudge

# ============================================================
# API CONFIGURATION - Add multiple keys for fallback
# ============================================================

# Judge (Gemini) API Keys
JUDGE_API_KEYS = [
    "AIzaSyAXvzzVFvrcdVZoMx7Q5PNXwxmkRS7rct8",
    # Add more Gemini keys here if needed:
    # "YOUR_BACKUP_GEMINI_KEY",
]

# Test Model (OpenRouter) API Keys - Multiple for fallback
TEST_API_KEYS = [
    "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",
    "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",
]

# Model Names
JUDGE_MODEL = "gemini-2.5-flash"
TEST_MODEL = "openai/gpt-oss-120b"

def main():
    print("=" * 70)
    print("GOLDEN 68 - RESILIENT EVALUATION FRAMEWORK")
    print("=" * 70)
    
    # Initialize adapters
    print("\n[1/5] Initializing adapters...")
    
    # Create resilient clients with fallback keys
    try:
        judge_client = ResilientModelClient(
            provider="gemini",
            model_name=JUDGE_MODEL,
            api_keys=JUDGE_API_KEYS
        )
        test_client = ResilientModelClient(
            provider="openrouter",
            model_name=TEST_MODEL,
            api_keys=TEST_API_KEYS
        )
        
        judge_adapter = judge_client._get_current_adapter()
        test_adapter = test_client._get_current_adapter()
        
        print(f"✓ Judge: Gemini 2.5-flash ({len(JUDGE_API_KEYS)} keys)")
        print(f"✓ Test: OpenRouter gpt-oss-120b ({len(TEST_API_KEYS)} keys)")
        
    except Exception as e:
        print(f"✗ Adapter initialization error: {e}")
        return
    
    # Quick test
    print("\n[2/5] Testing adapters...")
    try:
        test_response = test_adapter.generate("Hello")
        if "Error" not in test_response or test_adapter.is_credit_error(test_response):
            print(f"Test adapter: OK")
        else:
            print(f"Test adapter warning: {test_response[:50]}...")
    except APIKeyExhaustedError as e:
        print(f"✗ Test API exhausted: {e}")
        print("\n⚠️  Please provide additional OpenRouter API keys to continue.")
        return
    except Exception as e:
        print(f"Test adapter error: {e}")
    
    # Load dataset
    print("\n[3/5] Loading dataset...")
    loader = DatasetLoader()
    prompts = loader.get_all_prompts()
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Run evaluation
    print("\n[4/5] Running LLM-as-Judge evaluation...")
    llm_judge = LLMJudge(judge_adapter)
    results = []
    credit_errors = 0
    
    for i, prompt_data in enumerate(prompts):
        prompt_id = prompt_data.get("id", f"prompt_{i}")
        print(f"\n[{i+1}/{len(prompts)}] {prompt_id}...", end=" ", flush=True)
        
        try:
            # Get model response with automatic key rotation
            model_response = test_client.generate(
                prompt_data["prompt"],
                temperature=0.7
            )
            
            # Judge evaluation
            evaluation = llm_judge.evaluate(
                prompt=prompt_data["prompt"],
                model_response=model_response,
                expected_behavior=prompt_data.get("expected_behavior", ""),
                prompt_metadata=prompt_data
            )
            
            result = {
                "prompt_id": prompt_id,
                "pillar": prompt_data.get("pillar", ""),
                "level": prompt_data.get("level", 0),
                "category": prompt_data.get("category", ""),
                "prompt": prompt_data["prompt"],
                "expected_behavior": prompt_data.get("expected_behavior", ""),
                "model_response": model_response,
                "judge_score": evaluation.get("score", 0),
                "judge_determination": evaluation.get("determination", "UNKNOWN"),
                "judge_reasoning": evaluation.get("explanation", ""),
                "eu_act_ref": prompt_data.get("eu_act_ref", "")
            }
            results.append(result)
            
            score = evaluation.get("score", 0)
            det = evaluation.get("determination", "?")
            print(f"Score: {score}/10 ({det})")
            
        except APIKeyExhaustedError as e:
            print(f"\n\n{'='*70}")
            print("🔴 API KEY EXHAUSTED")
            print(f"{'='*70}")
            print(f"Provider: {e.provider}")
            print(f"Error: {e.message}")
            print(f"\nPrompts completed before failure: {len(results)}/{len(prompts)}")
            
            # Save partial results
            save_partial_results(results, judge_client, test_client)
            
            print("\n⚠️  TO CONTINUE TESTING:")
            print("   1. Add new API keys to the configuration above")
            print("   2.Restart the evaluation")
            print(f"\nPartial results saved. You can continue from prompt #{len(results)+1}")
            return
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "prompt_id": prompt_id,
                "pillar": prompt_data.get("pillar", ""),
                "level": prompt_data.get("level", 0),
                "error": str(e)
            })
        
        # Rate limiting
        time.sleep(0.3)
    
    # Calculate all statistics
    print("\n[5/5] Generating detailed report...")
    
    # Basic stats
    scores = [r.get("judge_score", 0) for r in results if "judge_score" in r]
    overall_score = sum(scores) / len(scores) if scores else 0
    passes = sum(1 for r in results if r.get("judge_determination") == "PASS")
    pass_rate = passes / len(results) if results else 0
    
    # Pillar breakdown
    pillars = defaultdict(lambda: {"scores": [], "passes": 0, "fails": 0, "results": []})
    for r in results:
        p = r.get("pillar", "Unknown")
        pillars[p]["results"].append(r)
        if "judge_score" in r:
            pillars[p]["scores"].append(r["judge_score"])
        if r.get("judge_determination") == "PASS":
            pillars[p]["passes"] += 1
        else:
            pillars[p]["fails"] += 1
    
    # Level breakdown
    levels = defaultdict(lambda: {"scores": [], "passes": 0})
    for r in results:
        lvl = r.get("level", 0)
        levels[lvl]["scores"].append(r.get("judge_score", 0))
        if r.get("judge_determination") == "PASS":
            levels[lvl]["passes"] += 1
    
    # Best and worst
    sorted_results = sorted([r for r in results if "judge_score" in r], 
                           key=lambda x: x["judge_score"], reverse=True)
    best_5 = sorted_results[:5]
    worst_5 = sorted_results[-5:]
    
    # Build detailed report
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "judge_model": JUDGE_MODEL,
            "test_model": TEST_MODEL,
            "judge_provider": "gemini",
            "test_provider": "openrouter",
            "judge_keys_count": len(JUDGE_API_KEYS),
            "test_keys_count": len(TEST_API_KEYS),
            "test_keys_used": test_client.current_key_index + 1
        },
        "summary": {
            "total_prompts": len(prompts),
            "evaluated": len(results),
            "overall_score": round(overall_score, 2),
            "pass_rate": round(pass_rate * 100, 1),
            "passes": passes,
            "grade": get_grade(overall_score)
        },
        "pillar_breakdown": {},
        "level_breakdown": {},
        "top_performers": best_5,
        "bottom_performers": worst_5,
        "all_results": results
    }
    
    # Pillar details
    for pillar, data in sorted(pillars.items()):
        avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        pr = data["passes"] / len(data["scores"]) * 100 if data["scores"] else 0
        report["pillar_breakdown"][pillar] = {
            "count": len(data["scores"]),
            "average_score": round(avg, 2),
            "pass_rate": round(pr, 1),
            "passes": data["passes"],
            "fails": data["fails"]
        }
    
    # Level details
    for lvl in sorted(levels.keys()):
        data = levels[lvl]
        avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        pr = data["passes"] / len(data["scores"]) * 100 if data["scores"] else 0
        report["level_breakdown"][f"level_{lvl}"] = {
            "count": len(data["scores"]),
            "average_score": round(avg, 2),
            "pass_rate": round(pr, 1),
            "passes": data["passes"]
        }
    
    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"detailed_evaluation_{timestamp}.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    
    print(f"""
OVERALL RESULTS:
  Total Prompts: {len(prompts)}
  Overall Score: {overall_score:.2f}/10 ({get_grade(overall_score)})
  Pass Rate: {pass_rate*100:.1f}%
  Passes: {passes}/{len(results)}

API STATUS:
  Judge Keys Used: {test_client.current_key_index + 1}/{len(JUDGE_API_KEYS)}
  Test Keys Used: {test_client.current_key_index + 1}/{len(TEST_API_KEYS)}

PILLAR BREAKDOWN:
""")
    
    for pillar, data in sorted(pillars.items()):
        avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        pr = data["passes"] / len(data["scores"]) * 100 if data["scores"] else 0
        status = "✓" if pr >= 50 else "✗"
        print(f"  {status} {pillar:15} Score: {avg:5.2f}/10 | Pass Rate: {pr:5.1f}% | {data['passes']}/{len(data['scores'])}")

    print(f"""
TOP 5 PERFORMERS:
""")
    for r in best_5:
        print(f"  ✓ {r['prompt_id']}: {r['judge_score']}/10 ({r['pillar']}, L{r['level']})")

    print(f"""
BOTTOM 5 PERFORMERS:
""")
    for r in worst_5:
        print(f"  ✗ {r['prompt_id']}: {r['judge_score']}/10 ({r['pillar']}, L{r['level']})")
    
    print(f"""
Results saved to: {output_file}
""")
    
    # Generate text report
    generate_text_report(report, output_dir, timestamp)
    
    return report


def save_partial_results(results, judge_client, test_client, output_dir=None):
    """Save partial results when API keys are exhausted."""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "data", "results")
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"partial_evaluation_{timestamp}.json")
    
    report = {
        "status": "PARTIAL",
        "timestamp": datetime.now().isoformat(),
        "prompts_completed": len(results),
        "judge_model": JUDGE_MODEL,
        "test_model": TEST_MODEL,
        "judge_keys_used": judge_client.current_key_index + 1,
        "test_keys_used": test_client.current_key_index + 1,
        "results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"Partial results saved to: {output_file}")


def get_grade(score: float) -> str:
    """Convert score to letter grade."""
    if score >= 9: return "A+"
    if score >= 8: return "A"
    if score >= 7: return "B"
    if score >= 6: return "C"
    if score >= 5: return "D"
    return "F"


def generate_text_report(report: dict, output_dir: str, timestamp: str):
    """Generate a detailed markdown report."""
    
    md = f"""# Golden 68 - Evaluation Report

**Generated:** {report['metadata']['timestamp']}  
**Judge Model:** {report['metadata']['judge_model']}  
**Test Model:** {report['metadata']['test_model']}  
**API Keys Used:** Judge: {report['metadata'].get('judge_keys_count', 1)}, Test: {report['metadata'].get('test_keys_count', 1)}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Prompts | {report['summary']['total_prompts']} |
| Evaluated | {report['summary']['evaluated']} |
| Overall Score | {report['summary']['overall_score']}/10 |
| Grade | {report['summary']['grade']} |
| Pass Rate | {report['summary']['pass_rate']}% |
| Prompts Passed | {report['summary']['passes']}/{report['summary']['evaluated']} |

---

## Pillar Analysis

"""
    
    for pillar, data in report["pillar_breakdown"].items():
        status = "✓" if data["pass_rate"] >= 50 else "✗"
        md += f"""### {status} {pillar}

| Metric | Value |
|--------|-------|
| Prompts | {data['count']} |
| Average Score | {data['average_score']}/10 |
| Pass Rate | {data['pass_rate']}% |
| Passed | {data['passes']} |
| Failed | {data['fails']} |

"""
    
    md += """---

## Level Analysis

| Level | Prompts | Avg Score | Pass Rate |
|-------|---------|-----------|-----------|
"""
    
    for level, data in sorted(report["level_breakdown"].items()):
        lvl = level.replace("level_", "")
        md += f"| {lvl} | {data['count']} | {data['average_score']}/10 | {data['pass_rate']}% |\n"
    
    md += """

---

## Top 5 Performing Prompts

"""
    for r in report["top_performers"]:
        md += f"""### ✓ {r['prompt_id']} - Score: {r['judge_score']}/10

**Pillar:** {r['pillar']} | **Level:** {r['level']}

**Prompt:** {r['prompt'][:200]}...

**Model Response:** {r['model_response'][:300]}...

**Judge Reasoning:** {r['judge_reasoning'][:500]}...

---

"""
    
    md += """

## Bottom 5 Performing Prompts

"""
    for r in report["bottom_performers"]:
        md += f"""### ✗ {r['prompt_id']} - Score: {r['judge_score']}/10

**Pillar:** {r['pillar']} | **Level:** {r['level']}

**Prompt:** {r['prompt'][:200]}...

**Model Response:** {r['model_response'][:300]}...

**Judge Reasoning:** {r['judge_reasoning'][:500]}...

---

"""
    
    md += f"""
---

*Report generated by Golden 68 Framework*
"""
    
    report_file = os.path.join(output_dir, f"report_{timestamp}.md")
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md)
    
    print(f"Markdown report saved to: {report_file}")


if __name__ == "__main__":
    main()
