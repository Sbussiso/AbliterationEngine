import json
d = json.load(open('/content/results/final_results.json'))
print(json.dumps(d, indent=1)[:3000])