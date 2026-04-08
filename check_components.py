"""Quick test script for all components"""
import sys
sys.path.insert(0, '.')

print("="*60)
print("GOLDEN 68 FRAMEWORK - COMPONENT CHECK")
print("="*60)

# 1. Test Dataset Loader
print("\n1. Testing DatasetLoader...")
try:
    from src.evaluation.loader import DatasetLoader
    loader = DatasetLoader()
    prompts = loader.get_all_prompts()
    print(f"   OK: {len(prompts)} prompts loaded")
    print(f"   Pillars: {loader.get_pillar_names()}")
except Exception as e:
    print(f"   ERROR: {e}")

# 2. Test Model Adapters
print("\n2. Testing Model Adapters...")
try:
    from src.models.adapters import ModelAdapterFactory, NVIDIAAdapter, APIKeyExhaustedError
    providers = ModelAdapterFactory.get_available_providers()
    print(f"   OK: Providers = {providers}")
    if "nvidia" in providers:
        print("   OK: NVIDIA adapter available")
except Exception as e:
    print(f"   ERROR: {e}")

# 3. Test LLM Judge
print("\n3. Testing LLMJudge...")
try:
    from src.judges.llm_judge import LLMJudge
    print("   OK: LLMJudge class available")
except Exception as e:
    print(f"   ERROR: {e}")

# 4. Test Human Audit
print("\n4. Testing HumanAudit...")
try:
    from src.audit.human_audit import HumanAuditManager, HumanAuditRecord
    manager = HumanAuditManager()
    print("   OK: HumanAuditManager available")
except Exception as e:
    print(f"   ERROR: {e}")

# 5. Test Report Generator
print("\n5. Testing ReportGenerator...")
try:
    from src.reporting.report_generator import ReportGenerator
    print("   OK: ReportGenerator available")
except Exception as e:
    print(f"   ERROR: {e}")

# 6. Test Streamlit App
print("\n6. Testing Streamlit App...")
try:
    import ast
    ast.parse(open('app.py').read())
    print("   OK: app.py syntax valid")
except Exception as e:
    print(f"   ERROR: {e}")

# 7. Test Trial Results
print("\n7. Checking Trial Test Results...")
try:
    import json
    with open('data/results/trial_test_20260323_113423.json') as f:
        results = json.load(f)
    print(f"   OK: {results['completed']}/{results['total']} prompts evaluated")
    print(f"   Avg Score: {results.get('avg_score', 0):.2f}/10")
    print(f"   Pass Rate: {results.get('passes', 0)}/{results['completed']}")
except Exception as e:
    print(f"   ERROR: {e}")

print("\n" + "="*60)
print("ALL COMPONENTS CHECKED!")
print("="*60)
