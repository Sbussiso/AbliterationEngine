import subprocess, os, json
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')

def colab_exec(code, timeout=60):
    p = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout',str(timeout)],
                       input=code, capture_output=True, text=True, timeout=timeout+30)
    return p.stdout

# check mmlu progress + base mmlu done?
out = colab_exec("""
import os, json, subprocess
print('== mmlu_results tree ==')
for root, dirs, files in os.walk('/content/mmlu_results'):
    for f in files:
        p = os.path.join(root, f)
        print(p, os.path.getsize(p))
print('== mmlu_log tail ==')
print(open('/content/mmlu_log.txt').read()[-3000:])
""", timeout=90)
print(out)