"""Check evaluation results"""
import json
import glob
import os

files = glob.glob(os.path.join("data", "results", "full_evaluation_*.json"))
files.sort(key=os.path.getmtime, reverse=True)

if files:
    with open(files[0], 'r') as f:
        data = json.load(f)
    
    print(f"File: {os.path.basename(files[0])}")
    print(f"Total prompts: {data['total_prompts']}")
    print(f"Evaluated: {data['evaluated']}")
    print(f"Overall score: {data['overall_score']:.2f}")
    print(f"Pass rate: {data['pass_rate']*100:.1f}%")
    print(f"Passes: {data['passes']}")
    
    print("\nFirst 5 results:")
    for r in data['results'][:5]:
        print(f"  {r['prompt_id']}: Score={r['judge_score']}, Det={r['judge_determination']}")
    
    print("\nLast 5 results:")
    for r in data['results'][-5:]:
        print(f"  {r['prompt_id']}: Score={r['judge_score']}, Det={r['judge_determination']}")
