import json, os
print("== exit codes ==")
for f in ['exit_code.txt','mmlu_exit_code.txt']:
    try:
        print(f, open('/content/'+f).read().strip())
    except Exception as e:
        print(f, 'ERR', e)
print("== run_log tail ==")
print(open('/content/run_log.txt').read()[-2500:])
print("== mmlu_log tail ==")
print(open('/content/mmlu_log.txt').read()[-2000:])
print("== ladder_log tail ==")
print(open('/content/ladder_log.txt').read()[-1500:])