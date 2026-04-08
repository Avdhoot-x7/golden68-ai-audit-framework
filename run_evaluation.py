"""
Run Full Evaluation Script for Golden 68 Framework
"""
import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.models.adapters import ModelAdapterFactory
from src.evaluation.loader import DatasetLoader
from src.judges.llm_judge import LLMJudge

# API Keys
JUDGE_API_KEY = "AIzaSyAXvzzVFvrcdVZoMx7Q5PNXwxmkRS7rct8"
TEST_API_KEY = "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"

# Model Names
JUDGE_MODEL = "gemini-2.5-flash"
TEST_MODEL = "openai/gpt-4o-mini"

def main():
    print("=" * 60)
    print("Golden 68 - Full Evaluation Runner")
    print("=" * 60)
    
    # Initialize adapters
    print("\n[1/4] Initializing adapters...")
    try:
        judge_adapter = ModelAdapterFactory.create("gemini", JUDGE_API_KEY, JUDGE_MODEL)
        test_adapter = ModelAdapterFactory.create("openrouter", TEST_API_KEY, TEST_MODEL)
        print("✓ Adapters initialized")
    except Exception as e:
        print(f"✗ Adapter error: {e}")
        return
    
    # Quick test
    print("\n[2/4] Testing adapters...")
    try:
        test_response = test_adapter.generate("Hello")
        print(f"Test adapter: {test_response[:50]}...")
    except Exception as e:
        print(f"✗ Test adapter error: {e}")
        return
    
    try:
        judge_test = judge_adapter.generate("Hello")
        print(f"Judge adapter: {judge_test[:50]}...")
    except Exception as e:
        print(f"✗ Judge adapter error: {e}")
        return
    
    # Load dataset
    print("\n[3/4] Loading dataset...")
    loader = DatasetLoader()
    prompts = loader.get_all_prompts()
    print(f"✓ Loaded {len(prompts)} prompts")
    
    # Run evaluation
    print("\n[4/4] Running evaluation...")
    llm_judge = LLMJudge(judge_adapter)
    results = []
    
    for i, prompt_data in enumerate(prompts):
        prompt_id = prompt_data.get("id", f"prompt_{i}")
        print(f"\n[{i+1}/{len(prompts)}] {prompt_id}...", end=" ", flush=True)
        
        try:
            # Get model response
            model_response = test_adapter.generate(
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
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "prompt_id": prompt_id,
                "pillar": prompt_data.get("pillar", ""),
                "level": prompt_data.get("level", 0),
                "error": str(e)
            })
        
        # Rate limiting
        time.sleep(0.5)
    
    # Save results
    print("\n" + "=" * 60)
    print("Saving results...")
    
    output_dir = os.path.join(os.path.dirname(__file__), "data", "results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"full_evaluation_{timestamp}.json")
    
    # Calculate summary
    scores = [r.get("judge_score", 0) for r in results if "judge_score" in r]
    overall_score = sum(scores) / len(scores) if scores else 0
    passes = sum(1 for r in results if r.get("judge_determination") == "PASS")
    pass_rate = passes / len(results) if results else 0
    
    summary = {
        "timestamp": timestamp,
        "judge_model": JUDGE_MODEL,
        "test_model": TEST_MODEL,
        "total_prompts": len(prompts),
        "evaluated": len(results),
        "overall_score": overall_score,
        "pass_rate": pass_rate,
        "passes": passes,
        "results": results
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  - Total Prompts: {len(prompts)}")
    print(f"  - Evaluated: {len(results)}")
    print(f"  - Overall Score: {overall_score:.2f}/10")
    print(f"  - Pass Rate: {pass_rate*100:.1f}%")
    print(f"  - Passes: {passes}/{len(results)}")
    
    # Print by pillar
    pillars = {}
    for r in results:
        p = r.get("pillar", "Unknown")
        if p not in pillars:
            pillars[p] = {"scores": [], "count": 0}
        if "judge_score" in r:
            pillars[p]["scores"].append(r["judge_score"])
        pillars[p]["count"] += 1
    
    print("\nBy Pillar:")
    for pillar, data in pillars.items():
        avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
        print(f"  - {pillar}: {avg:.2f}/10 ({len(data['scores'])}/{data['count']})")

if __name__ == "__main__":
    main()
