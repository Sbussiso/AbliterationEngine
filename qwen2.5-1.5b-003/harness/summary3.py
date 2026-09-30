import subprocess, os, json
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')

def colab_exec(code, timeout=60):
    p = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout',str(timeout)],
                       input=code, capture_output=True, text=True, timeout=timeout+30)
    return p.stdout

out = colab_exec("""
import json
rows = {}
for f in ['probes_baseline','probes_hook_ablated','probes_wd_B','probes_wd_BN','probes_wd_ML','probes_wd_ML_BN']:
    d = json.load(open('/content/abliteration_out/%s.json' % f))
    for kind in ['harmful','harmless']:
        for row in d[kind]:
            key = (kind, row['i'])
            rows.setdefault(key, {})[f] = row['refused']
# print matrix: for each probe index, refusal per condition
for kind in ['harmful','harmless']:
    print('###', kind)
    hdr = ['idx'] + ['base','hook','wdB','wdBN','wdML','wdMLBN']
    print(' | '.join(hdr))
    for i in range(16):
        line = [str(i)]
        for f in ['probes_baseline','probes_hook_ablated','probes_wd_B','probes_wd_BN','probes_wd_ML','probes_wd_ML_BN']:
            line.append(str(rows[(kind,i)].get(f,'-')))
        print(' | '.join(line))
print()
print('=== per-condition refusal rates (harmful) ===')
for f in ['probes_baseline','probes_hook_ablated','probes_wd_B','probes_wd_BN','probes_wd_ML','probes_wd_ML_BN']:
    d = json.load(open('/content/abliteration_out/%s.json' % f))
    h = sum(r['refused'] for r in d['harmful'])
    b = sum(r['refused'] for r in d['harmless'])
    dg = sum(r['degenerate'] for r in d['harmful']) + sum(r['degenerate'] for r in d['harmless'])
    print(f, 'refusal', h, '/16 =', h/16, 'benign_refusal', b, '/16 =', b/16, 'degenerate', dg)
""", timeout=90)
print(out)