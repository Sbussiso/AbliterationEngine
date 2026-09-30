import os, json
print("== /content ==")
for root, dirs, files in os.walk('/content'):
    # skip huge site-packages
    if 'site-packages' in root or 'dist-packages' in root:
        continue
    lvl = root.replace('/content','').count(os.sep)
    if lvl > 3:
        continue
    print(root, '->', sorted(files)[:40])