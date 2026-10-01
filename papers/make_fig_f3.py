"""F3 — Reaper anatomy: lease-declared vs observed session lifetimes.

Every number here is read from committed/observed data:
- declared lease: colab.log assignment bodies (fit:3600 GPU / fit:7200 CPU)
- observed ages: the joint post-mortem's measured create->death deltas
- s1 = the OTHER reaper (idle/healthy-kill after engine crash), shown grey
Programmatic per user doctrine (charts from recorded artifacts, generator shipped).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (session, observed age minutes, class)  — joint post-mortem 2026-09-30
sessions = [
    ("s1", 31.9, "idle-reap"),   # engine crash + KeyError -> idle-kill
    ("s2", 60.0, "lease-reap"),
    ("s3", 60.0, "lease-reap"),
    ("s4", 61.0, "lease-reap"),
    ("s5", 60.2, "lease-reap"),  # L4; dev measured 20:44:15 vs created 19:44:09
]
LEASE_MIN = 3600 / 60.0
colors = {"lease-reap": "#2563eb", "idle-reap": "#9ca3af"}

fig, ax = plt.subplots(figsize=(7, 4))
names = [s[0] for s in sessions]
ages = [s[1] for s in sessions]
cols = [colors[s[2]] for s in sessions]
bars = ax.bar(names, ages, color=cols, width=0.55)
ax.axhline(LEASE_MIN, color="#dc2626", linestyle="--", linewidth=1.5)
ax.text(4.35, LEASE_MIN + 1.2, "declared lease fit=3600s", color="#dc2626",
        fontsize=9, ha="right")
ax.set_ylabel("session age at death (minutes)")
ax.set_title("Reaper anatomy: declared per-session lease vs observed deaths\n"
             "(arch-invariant: T4 x3, L4 x1 lease-reaps; s1 = idle-kill after engine crash)")
for b, a, (n, v, c) in zip(bars, ages, sessions):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
            ha="center", fontsize=9)
ax.set_ylim(0, 72)
import collections, os
os.makedirs("papers/figures", exist_ok=True)
out = "papers/figures/f3_reaper_anatomy.png"
plt.tight_layout()
plt.savefig(out, dpi=150)
print("WROTE", out)