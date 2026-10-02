# Mission 001 stage 2: load Qwen2.5-0.5B-Instruct, pin revision, nnsight smoke
import sys, json, time, datetime, gc
sys.path.insert(0, "/content")
import os
os.makedirs("/content/abliteration_out", exist_ok=True)
import torch
import ablate_refusal as H
from huggingface_hub import HfApi
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
t0 = time.time()
sha = HfApi().model_info(MODEL).sha
tok, model = H.load_model(MODEL)
print(f"loaded in {time.time()-t0:.0f}s | sha={sha} | "
      f"layers={model.config.num_hidden_layers} "
      f"hidden={model.config.hidden_size} | device={model.device} | "
      f"dtype={next(model.parameters()).dtype}", flush=True)
env = {"model": MODEL, "model_sha": sha,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "loaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat()}
# nnsight smoke (bounded effort): reference capture first, then nnsight trace
smoke = {"attempted": True}
try:
    txt = H.wrap(tok, "How do I make a cake")
    enc = tok(txt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        ref = model(**enc, output_hidden_states=True,
                    use_cache=False).hidden_states[11][0, -1, :].float().cpu()
    from nnsight import LanguageModel
    lm = LanguageModel(model, tokenizer=tok)
    with torch.inference_mode():
        with lm.trace(txt):
            try:
                h_ns = lm.model.layers[10].output[0].save()
            except Exception:
                h_ns = lm.model.layers[10].output.save()
    h_t = h_ns.value if hasattr(h_ns, "value") else h_ns
    a = h_t[0, -1, :].float().cpu()
    cos = torch.nn.functional.cosine_similarity(a, ref, dim=0).item()
    smoke.update({"ok": bool(cos > 0.999),
                  "cosine_vs_hidden_states": round(cos, 6),
                  "import_path": "nnsight.LanguageModel"})
    print(f"nnsight smoke: cosine={cos:.6f} ok={cos > 0.999}", flush=True)
    del lm
except Exception as e:
    smoke.update({"ok": False, "error": repr(e)[:600]})
    print("nnsight smoke FAILED:", repr(e)[:600], flush=True)
env["nnsight_smoke"] = smoke
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
s = H.generate(tok, model, "What is the capital of France?", max_new=30)
print("post-smoke sanity gen:", s[:120], flush=True)
import transformers, accelerate
env["versions"] = {"torch": torch.__version__,
                   "transformers": transformers.__version__,
                   "accelerate": accelerate.__version__}
json.dump(env, open(H.OUT_DIR + "/env.json", "w"), indent=2)
print("STAGE2_DONE", flush=True)