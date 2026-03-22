import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

with open("results.json") as f:
    r = json.load(f)

tasks        = [1, 2, 3, 4, 5]
retrain_accs = r["retrain_accs"]
delta_accs   = r["delta_accs"]
naive_accs   = r["naive_accs"]
retrain_time = r["retrain_time"]
delta_time   = r["delta_time"]
naive_time   = r["naive_time"]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Delta-Only Model Training  vs  Full Retraining",
             fontsize=15, fontweight="bold", y=1.02)

# ── Chart 1: Accuracy over tasks ──────────────────────────
ax = axes[0]
ax.plot(tasks, retrain_accs, "o-",  color="#1A3A5C",
        lw=2.5, ms=8, label="Full retrain (oracle)")
ax.plot(tasks, delta_accs,  "s--", color="#1D9E75",
        lw=2.5, ms=8, label="Delta + EWC  ← your system")
ax.plot(tasks, naive_accs,  "^:",  color="#D85A30",
        lw=2,   ms=7, label="Naive delta (no memory)")

ax.fill_between(tasks, delta_accs, retrain_accs,
                alpha=0.12, color="#1D9E75")

for i, (r_, d_) in enumerate(zip(retrain_accs, delta_accs)):
    gap = abs(r_ - d_) * 100
    ax.annotate(f"{gap:.0f}%", xy=(tasks[i], (r_+d_)/2),
                ha="center", fontsize=8, color="#1D9E75")

ax.set_xlabel("Tasks seen", fontsize=11)
ax.set_ylabel("Average accuracy (all tasks)", fontsize=11)
ax.set_title("Accuracy vs Tasks", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0.5, 1.0)
ax.set_xticks(tasks)

# ── Chart 2: Training time bars ───────────────────────────
ax = axes[1]
methods = ["Full retrain", "Delta + EWC", "Naive delta"]
times   = [retrain_time, delta_time, naive_time]
colors  = ["#1A3A5C",    "#1D9E75",   "#D85A30"]
bars    = ax.bar(methods, times, color=colors, alpha=0.85, width=0.5)

for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 4,
            f"{t:.0f}s",
            ha="center", fontsize=11, fontweight="bold")

speedup = retrain_time / delta_time
ax.text(0.5, 0.92,
        f"Delta is {speedup:.1f}x faster",
        transform=ax.transAxes, ha="center",
        fontsize=12, fontweight="bold", color="#1D9E75",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="#D9EAD3", edgecolor="#1D9E75"))

ax.set_ylabel("Total training time (seconds)", fontsize=11)
ax.set_title("Compute Cost", fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(times) * 1.25)

# ── Chart 3: Gap over tasks ───────────────────────────────
ax = axes[2]
gaps_ewc   = [abs(r-d)*100 for r, d in zip(retrain_accs, delta_accs)]
gaps_naive = [abs(r-n)*100 for r, n in zip(retrain_accs, naive_accs)]

ax.plot(tasks, gaps_ewc,   "s--", color="#1D9E75",
        lw=2.5, ms=8, label="Delta + EWC gap")
ax.plot(tasks, gaps_naive, "^:",  color="#D85A30",
        lw=2,   ms=7, label="Naive delta gap")
ax.axhline(y=5, color="gray", linestyle="--",
           lw=1, alpha=0.6, label="5% target threshold")

ax.fill_between(tasks, gaps_ewc, gaps_naive,
                alpha=0.1, color="#1D9E75",
                label="EWC advantage")

ax.set_xlabel("Tasks seen", fontsize=11)
ax.set_ylabel("Gap vs full retrain (%)", fontsize=11)
ax.set_title("Accuracy Gap Over Time", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_ylim(0, 35)
ax.set_xticks(tasks)

plt.tight_layout()
plt.savefig("delta_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved to delta_results.png")