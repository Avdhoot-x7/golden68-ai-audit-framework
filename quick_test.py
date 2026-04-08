"""Quick test of adapters"""
import sys
sys.path.insert(0, '.')

from src.models.adapters import ModelAdapterFactory

print("Testing Gemini 2.5-flash...")
judge = ModelAdapterFactory.create("gemini", "AIzaSyAXvzzVFvrcdVZoMx7Q5PNXwxmkRS7rct8", "gemini-2.5-flash")
r = judge.generate("Say hello")
print(f"Result: {r}")

print("\nTesting OpenRouter...")
test = ModelAdapterFactory.create("openrouter", "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", "openai/gpt-4o-mini")
r = test.generate("Say hello")
print(f"Result: {r}")
