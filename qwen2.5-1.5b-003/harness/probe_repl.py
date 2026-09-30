import subprocess, ast, base64, os
r = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout','120'],
                   input='', capture_output=True, text=True, timeout=150)
print('rc', r.returncode)
print(r.stdout[-2000:])
print(r.stderr[-2000:])