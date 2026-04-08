"""Test API keys"""
import sys
sys.path.insert(0, '.')

from src.models.adapters import ModelAdapterFactory

keys = [
    "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",
    "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",
]

for i, key in enumerate(keys):
    print(f"\nTesting Key #{i+1}: {key[:30]}...")
    try:
        adapter = ModelAdapterFactory.create('openrouter', key, 'openai/gpt-oss-120b')
        r = adapter.generate("Say hello in one word")
        if "Error" not in r:
            print(f"✓ Key #{i+1} works: {r}")
        else:
            print(f"✗ Key #{i+1} error: {r[:100]}")
    except Exception as e:
        print(f"✗ Key #{i+1} exception: {e}")
