import base64, hashlib, os
names = ['probes_baseline.json','probes_hook_ablated.json','probes_wd_B.json','probes_wd_BN.json',
         'probes_wd_ML.json','probes_wd_ML_BN.json','layer_coherence.json','selection.json',
         'selection_candidates.json','run_config.json','refusal_direction.npy','refusal_direction_A.npy',
         'refusal_direction_B.npy','harness_sha256.json','ladder_sha256.json']
payload = {}
for n in names:
    p = '/content/abliteration_out/'+n
    payload[n] = base64.b64encode(open(p,'rb').read()).decode()
with open('/tmp/resp.txt','w') as f:
    f.write(repr(payload))
print('ready', sum(len(v) for v in payload.values()))