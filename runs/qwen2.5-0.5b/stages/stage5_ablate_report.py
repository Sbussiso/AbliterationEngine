# Mission 001 stage 5: ablate at best layer, probe, assemble report
import sys, json, time
sys.path.insert(0, "/content")
import numpy as np
import torch
import ablate_refusal as H
torch.manual_seed(0)
lc = json.load(open(H.OUT_DIR + "/layer_coherence.json"))
L = lc["best"]["decoder_layer"]
direction = torch.from_numpy(np.load(H.OUT_DIR + "/refusal_direction.npy"))
hook = H.AblationHook(direction, model.device, next(model.parameters()).dtype)
hook.attach(model.model.layers[L])
print(f"hook attached at model.model.layers[{L}]", flush=True)
t0 = time.time()
rows_h = H.run_probes(tok, model, H.HARMFUL[:16], tag="abl-harm")
rows_b = H.run_probes(tok, model, H.HARMLESS[:16], tag="abl-harmless")
hook.detach()
print(f"hook fired {hook.calls} times; detached ({time.time()-t0:.0f}s)", flush=True)
assert hook.calls > 0, "hook never fired - ablation invalid"
json.dump({"harmful": rows_h, "harmless": rows_b, "hook_calls": hook.calls},
          open(H.OUT_DIR + "/probes_ablated.json", "w"), indent=2)
base = json.load(open(H.OUT_DIR + "/probes_baseline.json"))
print("SAMPLE harmful BASELINE:", json.dumps(base["harmful"][0]["output"][:200]),
      flush=True)
print("SAMPLE harmful ABLATED :", json.dumps(rows_h[0]["output"][:200]), flush=True)
print("SAMPLE harmless BASELINE:", json.dumps(base["harmless"][0]["output"][:160]),
      flush=True)
print("SAMPLE harmless ABLATED :", json.dumps(rows_b[0]["output"][:160]), flush=True)
H.write_report(H.OUT_DIR)
rep = json.load(open(H.OUT_DIR + "/report.json"))
print("REPORT metrics:", json.dumps(rep["metrics"]), flush=True)
print("STAGE5_DONE", flush=True)