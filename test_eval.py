"""Quick test of evaluation with reasoning"""
import sys
sys.path.insert(0, '.')

from src.models.adapters import ModelAdapterFactory
from src.evaluation.loader import DatasetLoader
from src.judges.llm_judge import LLMJudge

print("Testing adapters...")
judge = ModelAdapterFactory.create('gemini', 'AIzaSyAXvzzVFvrcdVZoMx7Q5PNXwxmkRS7rct8', 'gemini-2.5-flash')
test = ModelAdapterFactory.create('openrouter', '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', 'openai/gpt-4o-mini')

loader = DatasetLoader()
prompts = loader.get_all_prompts()

print(f"Testing with first prompt: {prompts[0]['id']}")
p = prompts[0]
r = test.generate(p['prompt'], temperature=0.7)
print(f"Model response: {r[:150]}...")

llm_judge = LLMJudge(judge)
eval_result = llm_judge.evaluate(p['prompt'], r, p.get('expected_behavior', ''), p)
print(f"\nScore: {eval_result['score']}")
print(f"Determination: {eval_result['determination']}")
print(f"Reasoning length: {len(eval_result['explanation'])} chars")
print(f"Reasoning preview: {eval_result['explanation'][:300]}...")
