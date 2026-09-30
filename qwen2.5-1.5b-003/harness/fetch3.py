import subprocess, os, ast, base64, json, hashlib
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')
# On Colab: parse /tmp/resp.txt and split into per-file chunks in /tmp/chunks/
remote_code = r'''
import ast, base64, os, math
d = ast.literal_eval(open('/tmp/resp.txt').read())
os.makedirs('/tmp/chunks', exist_ok=True)
CH = 60000
manifest = []
for name, b64 in d.items():
    n_chunks = math.ceil(len(b64)/CH)
    for i in range(n_chunks):
        piece = b64[i*CH:(i+1)*CH]
        fn = f'/tmp/chunks/{i:03d}_{n_chunks:03d}_{name}.b64'
        open(fn,'w').write(piece)
        manifest.append(fn)
open('/tmp/chunks/manifest.json','w').write(json.dumps(manifest))
print('CHUNKED', len(manifest))
import json
'''
# json needed in remote; adjust: use json import
remote_code = remote_code.replace("import ast, base64, os, math", "import ast, base64, os, math, json")
p = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout','60'],
                   input=remote_code, capture_output=True, text=True, timeout=90)
print('chunk rc', p.returncode, '|', p.stdout[-300:])
assert 'CHUNKED' in p.stdout, p.stdout[-2000:]
# Now download chunks via `colab download`
manifest_r = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout','30'],
                   input="print(open('/tmp/chunks/manifest.json').read())", capture_output=True, text=True, timeout=60)
mline = [l for l in manifest_r.stdout.splitlines() if l.startswith('[')]
manifest = json.loads(mline[-1])
print('manifest entries:', len(manifest))
os.makedirs('/tmp/dl', exist_ok=True)
ok = 0
for fn in manifest:
    dst = '/tmp/dl/'+os.path.basename(fn)
    r = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','download','-s','m004-abl-rw-1',fn,'-o',dst],
                       capture_output=True, text=True, timeout=90)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        ok += 1
    else:
        print('FAIL', fn, r.stdout[-200:], r.stderr[-200:])
print('downloaded', ok, '/', len(manifest))