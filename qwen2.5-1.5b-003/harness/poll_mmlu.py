import subprocess, os, json
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')

def colab_exec(code, timeout=60):
    p = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout',str(timeout)],
                       input=code, capture_output=True, text=True, timeout=timeout+30)
    return p.stdout

out = colab_exec("""
print(open('/content/mmlu_exit_code.txt').read().strip())
print(open('/content/mmlu_log.txt').read()[-2500:])
""", timeout=60)
print(out)