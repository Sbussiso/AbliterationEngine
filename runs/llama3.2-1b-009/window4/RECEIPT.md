# Window 4 receipts (2026-10-03 ~22:00 PDT)

GPU-native wd_B weights pushed DIRECT from the T4 session to the private hub
under `gpu_native/` (model.safetensors 2,996,982,200 B + config + generation_config).

Byte-check result: hub etag of gpu_native/model.safetensors = cb9ad4d09bb787ac... 
≠ interim-main (CPU rebuild) etag 6417be31... (control: interim-main etag read
back exactly = local sha, so etags are true sha256 here → divergence is REAL).

Configs identical. Interpretation on the ledger: GPU-vs-CPU fp16 edit round-off
and/or per-window direction re-extraction last-bit variance; behavior identical
(8/64 digit-confirmed both windows; residuals 3-sig-fig agreement).

14/32 w2 chunk files (1.77 GB) were also banked this window before session #3
died; they are gitignored (*.bin) — inventory: chunks c00-c13 all 126,666,668 B.

Also: the derived single-variant spec did NOT take (runner.sh ran the bundled
spec) — full ladder re-ran instead; wd_B/wd_BN probes + variants banked, wd_ML
mid-run at lease end. Not a data problem.

git bytes omitted per *.bin repo hygiene; raw chunk copies live in
abliteration-runs/llama3.2-1b-009-window2/chunks/ on VM 151.
