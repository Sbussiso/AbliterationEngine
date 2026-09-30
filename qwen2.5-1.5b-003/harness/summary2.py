import subprocess, os, json
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')

def colab_exec(code, timeout=60):
    p = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout',str(timeout)],
                       input=code, capture_output=True, text=True, timeout=timeout+30)
    return p.stdout

out = colab_exec("""
import json
for f in ['probes_baseline','probes_hook_ablated','probes_wd_B','probes_wd_BN','probes_wd_ML','probes_wd_ML_BN']:
    d = json.load(open('/content/abliteration_out/%s.json' % f))
    print('==', f, '==')
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                print(' ', k, '=', v)
            else:
                print(' ', k, ':', type(v).__name__, str(v)[:200])
    else:
        print(str(d)[:500])
""")
print(out)