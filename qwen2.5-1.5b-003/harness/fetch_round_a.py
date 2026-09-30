import subprocess, os, ast, base64, json, hashlib, shutil
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')

def colab_exec(code, timeout=60):
    p = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout',str(timeout)],
                       input=code, capture_output=True, text=True, timeout=timeout+30)
    return p.stdout

# Round A: print b64 in slices with markers, collect here.
names = ['probes_baseline.json','probes_hook_ablated.json','probes_wd_B.json','probes_wd_BN.json',
         'probes_wd_ML.json','probes_wd_ML_BN.json','layer_coherence.json','selection.json',
         'selection_candidates.json','run_config.json','refusal_direction.npy','refusal_direction_A.npy',
         'refusal_direction_B.npy','harness_sha256.json','ladder_sha256.json']
# Build one code cell that prints each file's b64 in labeled segments of 40k chars, with per-line prefixes.
code = r'''
import base64, os
names = {names_placeholder}
out = {}
for n in names:
    p = '/content/abliteration_out/' + n
    b64 = base64.b64encode(open(p,'rb').read()).decode()
    out[n] = b64
import json
blob = json.dumps(out)  # one big JSON string
print('BLOB_START')
for i in range(0, len(blob), 40000):
    print(blob[i:i+40000])
print('BLOB_END')
'''
code = code.replace('{names_placeholder}', repr(names))
out = colab_exec(code, timeout=180)
assert 'BLOB_START' in out and 'BLOB_END' in out, out[-3000:]
s = out.index('BLOB_START') + len('BLOB_START')
e = out.rindex('BLOB_END')
blob = out[s:e].strip()
blob = ''.join(blob.split())  # strip whitespace/newlines the kernel may wrap
d = json.loads(blob)
print('round A got', len(d), 'files, total b64', sum(len(v) for v in d.values()))
os.makedirs('/root/research/abliteration/qwen2.5-1.5b-003/artifacts/abliteration_out', exist_ok=True)
ad = '/root/research/abliteration/qwen2.5-1.5b-003/artifacts/abliteration_out'
for n, b64 in d.items():
    raw = base64.b64decode(b64)
    with open(os.path.join(ad, n), 'wb') as f:
        f.write(raw)
    print('saved', n, len(raw), hashlib.sha256(raw).hexdigest()[:16])
print('ROUND_A_DONE')