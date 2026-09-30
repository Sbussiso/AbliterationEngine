import subprocess, os, ast, base64, json, hashlib, shutil
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')
os.makedirs('/tmp/dl', exist_ok=True)
shutil.rmtree('/tmp/dl', ignore_errors=True); os.makedirs('/tmp/dl')

def colab_exec(code, timeout=60):
    p = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout',str(timeout)],
                       input=code, capture_output=True, text=True, timeout=timeout+30)
    return p.stdout

out = colab_exec("import json; print(open('/tmp/chunks/manifest.json').read())")
lines = [l for l in out.splitlines() if l.strip().startswith('[')]
manifest = json.loads(lines[-1])
print('manifest entries:', len(manifest))

def colab_download(remote, local):
    return subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','download','-s','m004-abl-rw-1',remote,local],
                          capture_output=True, text=True, timeout=120)

ok = 0
for fn in manifest:
    dst = '/tmp/dl/'+os.path.basename(fn)
    r = colab_download(fn, dst)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        ok += 1
    else:
        print('FAIL', fn, r.stdout[-300:], r.stderr[-300:])
print('downloaded', ok, '/', len(manifest))