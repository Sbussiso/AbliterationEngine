import subprocess, os, ast, base64, json, hashlib
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')
manifest = json.loads(subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout','30'],
                   input="import json; print(open('/tmp/chunks/manifest.json').read())", capture_output=True, text=True, timeout=60).stdout.split('[')[-1].rsplit(']')[0].join('[]'))