"""F3 card chart: capability guardrail — MMLU base vs variant vs the 3.0pp limit,
digits read from the banked mmlu_summary.json (never hand-typed)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
m = json.load(open('/root/research/abliteration-runs/qwen2.5-0.5b-002/eng_run002_pull_s5/mmlu_summary.json'))
base = m['base']['acc']*100; var = m['variant_model']['acc']*100
sb = m['base']['acc_stderr']*100; sv = m['variant_model']['acc_stderr']*100
delta = base - var; limit = m['guardrail_loss_pp_limit']; margin = limit/delta
plt.rcParams.update({
    "figure.facecolor": "#131417", "axes.facecolor": "#131417",
    "savefig.facecolor": "#131417", "text.color": "#e8e4da",
    "axes.edgecolor": "#3a3d44", "axes.labelcolor": "#e8e4da",
    "xtick.color": "#e8e4da", "ytick.color": "#e8e4da",
    "grid.color": "#26282e", "font.family": "DejaVu Sans", "font.size": 10,
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.4), dpi=160,
                               gridspec_kw={'width_ratios': [1.15, 1]})
bars = ax1.bar(['base Qwen2.5-1.5B', 'variant wd_ML_BN'], [base, var],
               color=['#6b6b6b', '#c8a24a'], width=0.52)
ax1.errorbar([0, 1], [base, var], yerr=[sb, sv], fmt='none', ecolor='#e8e4da',
             capsize=4, lw=1.2)
for x, v, s in ((0, base, sb), (1, var, sv)):
    ax1.text(x, v + s + 0.25, f'{v:.2f}%', ha='center', fontsize=10)
ax1.set_ylim(55, 63)
ax1.set_ylabel('MMLU accuracy (%, 0-shot, subject-weighted)')
ax1.set_title('Run 002 guardrail: capability retention', fontsize=11, pad=10)
ax1.grid(axis='y', alpha=.5); ax1.spines[['top','right']].set_visible(False)
ax2.bar(['observed loss', 'guardrail limit'], [delta, limit],
        color=['#3f6d8e', '#d96f5c'], width=0.52)
ax2.text(0, delta + 0.08, f'{delta:.2f}pp', ha='center', fontsize=10)
ax2.text(1, limit + 0.08, f'{limit:.1f}pp  ({margin:.0f}× margin)', ha='center', fontsize=10)
ax2.set_ylim(0, 3.6)
ax2.set_ylabel('MMLU loss (pp)')
ax2.set_title('loss vs the designed limit', fontsize=11, pad=10)
ax2.grid(axis='y', alpha=.5); ax2.spines[['top','right']].set_visible(False)
fig.tight_layout(); fig.savefig('/tmp/rebuild15b/f3_mmlu_guardrail.png')
print('F3 computed from artifact: base=%.6f var=%.6f delta=%.4fpp margin=%.1f' % (base, var, delta, margin))
