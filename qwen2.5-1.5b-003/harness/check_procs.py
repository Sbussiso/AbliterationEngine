import subprocess, os, json
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')

def colab_exec(code, timeout=60):
    p = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout',str(timeout)],
                       input=code, capture_output=True, text=True, timeout=timeout+30)
    return p.stdout

out = colab_exec("""
import subprocess, os
# is lm_eval still running?
r = subprocess.run(['bash','-c','ps aux | grep -E "lm_eval|python3 -m" | grep -v grep || true'], capture_output=True, text=True)
print('== procs ==')
print(r.stdout)
# GPU utilization
r2 = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print('== nvidia-smi ==')
print(r2.stdout[:1500])
""", timeout=90)
print(out)