# Mission 001 stage 2b: recover env.json after nnsight smoke crash (kernel state persists)
import sys, json, gc, datetime
sys.path.insert(0, "/content")
import torch
import ablate_refusal as H
from huggingface_hub import HfApi
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
sha = HfApi().model_info(H.env_model if hasattr(H, "env_model") else "Qwen/Qwen2.5-0.5B-Instruct").sha
env = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "model_sha": sha,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
       "loaded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
       "nnsight_smoke": {
           "attempted": True, "ok": False,
           "error": ("nnsight 0.7.0 LanguageModel wrapper on accelerate-dispatched "
                     "fp16 Qwen2.5: ExitTracingException raised from inside "
                     "lm.trace() body at layers[10].output[0].save(); exception is "
                     "BaseException-derived and escapes `except Exception`; "
                     "fell back to output_hidden_states=True backend per playbook"),
           "import_path": "nnsight.LanguageModel (import OK, trace FAIL)"}}
import transformers, accelerate
env["versions"] = {"torch": torch.__version__,
                   "transformers": transformers.__version__,
                   "accelerate": accelerate.__version__}
s = H.generate(tok, model, "What is the capital of France?", max_new=30)
print("sanity gen:", s[:120], flush=True)
json.dump(env, open(H.OUT_DIR + "/env.json", "w"), indent=2)
print("STAGE2B_DONE env.json written", flush=True)