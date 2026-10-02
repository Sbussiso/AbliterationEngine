# Mission 001 stage 3: capture all layers, coherence scan, pick layer, save direction
import sys, json, time
sys.path.insert(0, "/content")
import numpy as np
import ablate_refusal as H
t0 = time.time()
print("capturing harmful activations (64 prompts)...", flush=True)
cap_harm = H.capture_all_layers(tok, model,
                                [H.wrap(tok, p) for p in H.HARMFUL], batch=16)
print("capturing harmless activations (64 prompts)...", flush=True)
cap_harmless = H.capture_all_layers(tok, model,
                                    [H.wrap(tok, p) for p in H.HARMLESS], batch=16)
print(f"capture done in {time.time()-t0:.0f}s", flush=True)
best, table, direction = H.scan_layers(cap_harm, cap_harmless, n_pairs=64)
print(f"BEST decoder_layer={best['decoder_layer']} "
      f"coherence={best['coherence']} |d|={best['direction_norm']}", flush=True)
np.save(H.OUT_DIR + "/refusal_direction.npy", direction.numpy())
json.dump({"best": best, "table": table, "n_pairs": 64},
          open(H.OUT_DIR + "/layer_coherence.json", "w"), indent=2)
print("STAGE3_DONE", flush=True)