# Mission 001 stage 4: baseline (clean model) probes
import sys, json, time
sys.path.insert(0, "/content")
import torch
import ablate_refusal as H
torch.manual_seed(0)
t0 = time.time()
rows_h = H.run_probes(tok, model, H.HARMFUL[:16], tag="base-harm")
rows_b = H.run_probes(tok, model, H.HARMLESS[:16], tag="base-harmless")
json.dump({"harmful": rows_h, "harmless": rows_b},
          open(H.OUT_DIR + "/probes_baseline.json", "w"), indent=2)
print(f"BASELINE refusal_rate={H.refusal_rate(rows_h):.3f} "
      f"benign_preserved={1 - H.refusal_rate(rows_b):.3f} "
      f"({time.time()-t0:.0f}s)", flush=True)
print("STAGE4_DONE", flush=True)