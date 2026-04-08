"""Test Gemini as both judge and test model"""
import sys
sys.path.insert(0, '.')

from src.models.adapters import ModelAdapterFactory
from src.evaluation.loader import DatasetLoader
from src.judges.llm_judge import LLMJudge

print("Testing Gemini 2.5-flash for both roles...")
gemini = ModelAdapterFactory.create('gemini', 'AIzaSyAXvzzVFvrcdVZoMx7Q5PNXwxmkRS7rct8', 'gemini-2.5-flash')

loader = DatasetLoader()
prompts = loader.get_all_prompts()

print(f"Testing with first prompt: {prompts[0]['id']}")
p = prompts[0]

# Test model response
print("\n--- Test Model Response ---")
r = gemini.generate(p['prompt'], temperature=0.7)
print(f"Response: {r[:300]}...")

# Judge evaluation
print("\n--- Judge Evaluation ---")
llm_judge = LLMJudge(gemini)
eval_result = llm_judge.evaluate(p['prompt'], r, p.get('expected_behavior', ''), p)
print(f"Score: {eval_result['score']}")
print(f"Determination: {eval_result['determination']}")
print(f"Reasoning: {eval_result['explanation'][:400]}...")
