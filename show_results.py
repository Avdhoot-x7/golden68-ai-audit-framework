import json
import glob
import os
from collections import defaultdict

files = glob.glob("data/results/detailed_evaluation_*.json")
latest = max(files, key=os.path.getmtime)

with open(latest, 'r') as f:
    d = json.load(f)

print("=" * 70)
print("GOLDEN 68 - COMPREHENSIVE EVALUATION RESULTS")
print("=" * 70)
print(f"\nModels:")
print(f"  Judge: {d['metadata']['judge_model']}")
print(f"  Test:  {d['metadata']['test_model']}")
print(f"\nOverall Results:")
print(f"  Total Prompts: {d['summary']['total_prompts']}")
print(f"  Overall Score: {d['summary']['overall_score']}/10 ({d['summary']['grade']})")
print(f"  Pass Rate: {d['summary']['pass_rate']}%")
print(f"  Passes: {d['summary']['passes']}/{d['summary']['evaluated']}")

# Pillar breakdown
print("\n" + "-" * 50)
print("PILLAR BREAKDOWN")
print("-" * 50)

for pillar, data in d['pillar_breakdown'].items():
    status = "✓" if data["pass_rate"] >= 50 else "✗"
    print(f"\n{status} {pillar}:")
    print(f"  Prompts: {data['count']}")
    print(f"  Average Score: {data['average_score']}/10")
    print(f"  Pass Rate: {data['pass_rate']}% ({data['passes']}/{data['count']})")

# Level breakdown
print("\n" + "-" * 50)
print("LEVEL BREAKDOWN")
print("-" * 50)

for level, data in sorted(d['level_breakdown'].items()):
    lvl = level.replace("level_", "")
    print(f"Level {lvl}: {data['average_score']}/10 | Pass Rate: {data['pass_rate']}%")

# Top/Bottom
print("\n" + "-" * 50)
print("TOP 5 PERFORMING PROMPTS")
print("-" * 50)
for r in d['top_performers']:
    print(f"  ✓ {r['prompt_id']}: {r['judge_score']}/10 ({r['pillar']}, L{r['level']})")

print("\n" + "-" * 50)
print("BOTTOM 5 PERFORMING PROMPTS")
print("-" * 50)
for r in d['bottom_performers']:
    print(f"  ✗ {r['prompt_id']}: {r['judge_score']}/10 ({r['pillar']}, L{r['level']})")

# Sample reasoning
print("\n" + "-" * 50)
print("SAMPLE JUDGE REASONING (First Pass)")
print("-" * 50)
for r in d['all_results']:
    if r['judge_determination'] == 'PASS' and r.get('judge_reasoning'):
        print(f"\nPrompt: {r['prompt_id']}")
        print(f"Judge Score: {r['judge_score']}/10 - {r['judge_determination']}")
        print(f"Reasoning: {r['judge_reasoning'][:500]}...")
        break

print("\n" + "=" * 70)
print(f"Results: {latest}")
print(f"Report: {latest.replace('detailed_evaluation_', 'report_').replace('.json', '.md')}")
print("=" * 70)
