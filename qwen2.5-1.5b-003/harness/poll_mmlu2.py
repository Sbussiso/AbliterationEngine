import subprocess, os, json
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')

def colab_exec(code, timeout=60):
    p = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout',str(timeout)],
                       input=code, capture_output=True, text=True, timeout=timeout+30)
    return p.stdout

out = colab_exec("""
print('mmlu_exit_code:', open('/content/mmlu_exit_code.txt').read().strip())
import os, glob
files = sorted(glob.glob('/content/mmlu_results/**/results_*.json', recursive=True))
print('results files:', files)
for f in files:
    print('==', f)
    print(open(f).read()[:800])
print('== summary file?')
print(os.path.exists('/content/mmlu_summary.json'))
if os.path.exists('/content/mmlu_summary.json'):
    print(open('/content/mmlu_summary.json').read())
""", timeout=90)
print(out[:6000])