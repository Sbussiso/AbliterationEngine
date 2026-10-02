# Mission 001 stage 1: GPU check + install deps + nnsight import probe
import subprocess, sys
import torch
print("GPU:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
      flush=True)
print("torch(before):", torch.__version__, flush=True)
pkgs = ["transformers", "accelerate", "nnsight", "datasets", "huggingface_hub"]
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U"] + pkgs,
                   capture_output=True, text=True)
print("pip rc:", r.returncode, flush=True)
if r.returncode != 0:
    print("pip stderr tail:", r.stderr[-2500:], flush=True)
print("torch(after):", torch.__version__,
      "cuda_ok:", torch.cuda.is_available(), flush=True)
import transformers, accelerate
print("transformers:", transformers.__version__,
      "accelerate:", accelerate.__version__, flush=True)
try:
    import nnsight
    print("nnsight:", nnsight.__version__, flush=True)
    try:
        from nnsight import LanguageModel
        print("nnsight LanguageModel import: nnsight.LanguageModel OK", flush=True)
    except ImportError:
        try:
            from nnsight.modeling.language import LanguageModel
            print("nnsight LanguageModel import: nnsight.modeling.language OK",
                  flush=True)
        except ImportError:
            print("nnsight LanguageModel import: FAILED (both paths)", flush=True)
except Exception as e:
    print("nnsight import failed:", repr(e)[:300], flush=True)
print("STAGE1_DONE", flush=True)