import subprocess, os, json
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')

def colab_exec(code, timeout=60):
    p = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout',str(timeout)],
                       input=code, capture_output=True, text=True, timeout=timeout+30)
    return p.stdout

out = colab_exec("""
import json
r = json.load(open('/content/abliteration_out/run_config.json'))
print(json.dumps(r, indent=1))
print('=== selection ===')
print(open('/content/abliteration_out/selection.json').read())
print('=== candidates ===')
print(open('/content/abliteration_out/selection_candidates.json').read())
print('=== coherence ===')
print(open('/content/abliteration_out/layer_coherence.json').read())
print('=== harness sha ===')
print(open('/content/abliteration_out/harness_sha256.json').read())
print('=== ladder sha ===')
print(open('/content/abliteration_out/ladder_sha256.json').read())
""")
print(out)