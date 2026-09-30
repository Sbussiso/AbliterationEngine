import json, base64, os, hashlib

def b64d(path):
    with open(path, 'rb') as f:
        return base64.b64encode(f.read()).decode()

def save_b64(path, b64):
    with open(path, 'wb') as f:
        f.write(base64.b64decode(b64))

out = {}
names = ['probes_baseline.json','probes_hook_ablated.json','probes_wd_B.json','probes_wd_BN.json',
         'probes_wd_ML.json','probes_wd_ML_BN.json','layer_coherence.json','selection.json',
         'selection_candidates.json','run_config.json']
for n in names:
    p = '/content/abliteration_out/'+n
    if os.path.exists(p):
        out[n] = open(p).read()
print(json.dumps({'files': {k: len(v) for k, v in out.items()}, 'missing': [n for n in names if n not in out]}))