import json
import glob
import os
from collections import defaultdict

files = glob.glob("data/results/*.json")
latest = max(files, key=os.path.getmtime)

with open(latest, 'r') as f:
    d = json.load(f)

print("=" * 60)
print("GOLDEN 68 - FULL EVALUATION RESULTS")
print("=" * 60)
print(f"\nModels:")
print(f"  Judge: {d['judge_model']}")
print(f"  Test:  {d['test_model']}")
print(f"\nOverall Results:")
print(f"  Total Prompts: {d['total_prompts']}")
print(f"  Overall Score: {d['overall_score']:.2f}/10")
print(f"  Pass Rate: {d['pass_rate']*100:.1f}%")
print(f"  Passes: {d['passes']}/{d['total_prompts']}")

# Pillar breakdown
print("\n" + "-" * 40)
print("PILLAR BREAKDOWN")
print("-" * 40)

pillars = defaultdict(lambda: {"scores": [], "passes": 0, "fails": 0})
for r in d['results']:
    p = r.get('pillar', 'Unknown')
    pillars[p]["scores"].append(r['judge_score'])
    if r['judge_determination'] == 'PASS':
        pillars[p]["passes"] += 1
    else:
        pillars[p]["fails"] += 1

for pillar, data in sorted(pillars.items()):
    avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
    pr = data["passes"] / len(data["scores"]) * 100 if data["scores"] else 0
    print(f"\n{pillar}:")
    print(f"  Prompts: {len(data['scores'])}")
    print(f"  Average Score: {avg:.2f}/10")
    print(f"  Pass Rate: {pr:.1f}% ({data['passes']}/{len(data['scores'])})")

# Level breakdown
print("\n" + "-" * 40)
print("COMPLEXITY LEVEL BREAKDOWN")
print("-" * 40)

levels = defaultdict(lambda: {"scores": [], "passes": 0})
for r in d['results']:
    lvl = r.get('level', 0)
    levels[lvl]["scores"].append(r['judge_score'])
    if r['judge_determination'] == 'PASS':
        levels[lvl]["passes"] += 1

for lvl in sorted(levels.keys()):
    data = levels[lvl]
    avg = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
    pr = data["passes"] / len(data["scores"]) * 100 if data["scores"] else 0
    print(f"Level {lvl}: {avg:.2f}/10, Pass Rate: {pr:.1f}%")

# Best and worst
print("\n" + "-" * 40)
print("TOP 5 PERFORMING PROMPTS")
print("-" * 40)

sorted_results = sorted(d['results'], key=lambda x: x['judge_score'], reverse=True)
for r in sorted_results[:5]:
    print(f"  {r['prompt_id']}: {r['judge_score']}/10 ({r['pillar']}, L{r['level']})")

print("\n" + "-" * 40)
print("BOTTOM 5 PERFORMING PROMPTS")
print("-" * 40)

for r in sorted_results[-5:]:
    print(f"  {r['prompt_id']}: {r['judge_score']}/10 ({r['pillar']}, L{r['level']})")

# Sample judge reasoning
print("\n" + "-" * 40)
print("SAMPLE JUDGE REASONING (First Pass)")
print("-" * 40)

for r in d['results']:
    if r['judge_determination'] == 'PASS':
        print(f"\nPrompt: {r['prompt_id']}")
        print(f"Model Response: {r['model_response'][:200]}...")
        print(f"\nJudge: {r['judge_reasoning'][:300]}...")
        break

print("\n" + "=" * 60)
print(f"Results saved to: {os.path.basename(latest)}")
print("=" * 60)
