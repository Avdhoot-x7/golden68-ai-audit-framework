import json

with open('data/results/detailed_evaluation_20260322_120225.json') as f:
    d = json.load(f)

print('='*60)
print('GOLDEN 68 - COMPLETE RESULTS')
print('='*60)
print(f"Judge: {d['metadata']['judge_model']}")
print(f"Test: {d['metadata']['test_model']}")
print()
print('SUMMARY:')
print(f"  Overall Score: {d['summary']['overall_score']}/10 ({d['summary']['grade']})")
print(f"  Pass Rate: {d['summary']['pass_rate']}%")
print(f"  Passed: {d['summary']['passes']}/{d['summary']['total_prompts']}")
print()
print('PILLAR BREAKDOWN:')
for p, data in d['pillar_breakdown'].items():
    print(f"  {p}: {data['average_score']}/10 | {data['pass_rate']}% pass")
print()
print('TOP 5:')
for r in d['top_performers']:
    print(f"  {r['prompt_id']}: {r['judge_score']}/10 - {r['judge_determination']}")
print()
print('BOTTOM 5:')
for r in d['bottom_performers']:
    print(f"  {r['prompt_id']}: {r['judge_score']}/10")

# Show sample reasoning
print()
print('='*60)
print('SAMPLE JUDGE REASONING (First PASS):')
print('='*60)
for r in d['all_results']:
    if r['judge_determination'] == 'PASS' and r.get('judge_reasoning'):
        print(f"\nPrompt: {r['prompt_id']}")
        print(f"Score: {r['judge_score']}/10 - {r['judge_determination']}")
        print(f"\nReasoning:\n{r['judge_reasoning'][:800]}...")
        break
