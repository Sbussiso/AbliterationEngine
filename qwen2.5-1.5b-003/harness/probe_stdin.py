import subprocess, os, time, json
os.chdir('/root/research/abliteration/qwen2.5-1.5b-003/harness')
code = "import ast, base64, json; d = ast.literal_eval(open('/tmp/resp.txt').read()); print('PARSED_OK', len(d), sum(len(v) for v in d.values()))"
# feed via stdin to `colab repl` won't work non-interactively; instead try `colab exec` with stdin file: the exec --file reads a file; we can't push /tmp/resp.txt as code because 260KB.
# Test: does `colab exec` accept stdin when no -f and empty stdin means nothing? Use pty-free approach: write code that reads from a file we upload first via FILE upload mechanism? colab CLI has no upload cmd in help. Try tiny probe.
p = subprocess.run(['sudo','-u','sbussiso','/home/sbussiso/.local/bin/colab','exec','-s','m004-abl-rw-1','--timeout','30'],
                   input=code, capture_output=True, text=True, timeout=60)
print('rc', p.returncode)
print('STDOUT:', p.stdout[-1500:])
print('STDERR:', p.stderr[-500:])