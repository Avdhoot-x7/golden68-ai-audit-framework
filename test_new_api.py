"""Test new API key"""
import sys
sys.path.insert(0, '.')

from src.models.adapters import ModelAdapterFactory
from src.evaluation.loader import DatasetLoader
from src.judges.llm_judge import LLMJudge

print("Testing NEW OpenRouter API key with gpt-oss-120b...")
test = ModelAdapterFactory.create(
    'openrouter', 
    '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', 
    'openai/gpt-oss-120b'
)

# Test
r = test.generate("Say hello in one word")
print(f"Response: {r}")

if "Error" not in r:
    print("\n✓ New API key works!")
else:
    print("\n✗ New API key failed!")
