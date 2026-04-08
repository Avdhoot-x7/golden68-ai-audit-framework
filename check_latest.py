import json
import glob
import os

files = glob.glob("data/results/*.json")
latest = max(files, key=os.path.getmtime)

with open(latest, 'r') as f:
    d = json.load(f)

print(f"File: {os.path.basename(latest)}")
print(f"Total: {d['total_prompts']}, Evaluated: {d['evaluated']}")
print(f"Score: {d['overall_score']:.2f}, Passes: {d['passes']}")
print(f"Pass Rate: {d['pass_rate']*100:.1f}%")
print("\nLast 5:")
for r in d['results'][-5:]:
    print(f"  {r['prompt_id']}: Score={r['judge_score']}, Det={r['judge_determination']}")
