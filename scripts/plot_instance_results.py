"""Regenerate multi-instance benchmark plots from saved CSV."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "results" / "gate" / "multi_instance" / "run1"
df      = pd.read_csv(RES_DIR / "benchmark_instances_results.csv")
df_sum  = pd.read_csv(RES_DIR / "benchmark_instances_summary.csv")

cities = df_sum["city"].tolist()
N_vals = df_sum["N"].tolist()
x      = np.arange(len(cities))
width  = 0.2

COLORS = {"no_gate":"#2471A3","fixed_gate":"#C0392B",
          "rolling":"#27AE60","calibration":"#E67E22"}
col_map = {"no_gate":"no_gate","fixed_gate":"fixed",
           "rolling":"rolling","calibration":"calib"}
LABELS  = {"no_gate":"No gate","fixed_gate":"Fixed",
           "rolling":"Rolling","calibration":"Calib a=0.01"}
METHODS = ["no_gate","fixed_gate","rolling","calibration"]

plt.rcParams.update({
    "font.family":"serif","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,
})

# ── Plot 1: 3-panel summary ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(
    "Multi-instance benchmark   seed=11   alpha=0.01   floor=0.90",
    fontsize=12, y=1.02,
)

# Left: grouped bars — absolute best score (normalised per city for readability)
ax = axes[0]
offsets = [-1.5, -0.5, 0.5, 1.5]
for mi, method in enumerate(METHODS):
    vals = df_sum[col_map[method]].tolist()
    ax.bar(x + offsets[mi]*width, vals, width*0.9,
           color=COLORS[method], alpha=0.8, label=LABELS[method], edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\nN={n}" for c, n in zip(cities, N_vals)], fontsize=8)
ax.set_ylabel("Best UE score", fontsize=11)
ax.set_title("Best score by city and method", fontsize=10, pad=6)
ax.legend(fontsize=9, frameon=False)
ax.grid(axis="y", alpha=0.2, linestyle="--")

# Middle: delta calib vs rolling
ax2 = axes[1]
deltas = df_sum["delta_calib_roll"].tolist()
bar_c  = ["#27AE60" if d > 0.05 else ("#F39C12" if abs(d) <= 0.05 else "#C0392B")
          for d in deltas]
ax2.bar(x, deltas, color=bar_c, edgecolor="white", linewidth=0.8)
ax2.axhline(0, color="#555", linewidth=1.2)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{c}\nN={n}" for c, n in zip(cities, N_vals)], fontsize=8)
ax2.set_ylabel("delta  calib - rolling", fontsize=11)
ax2.set_title("Calibration gain over rolling\nby instance size", fontsize=10, pad=6)
ax2.grid(axis="y", alpha=0.2, linestyle="--")
for xi, d in enumerate(deltas):
    ax2.text(xi, d + (0.05 if d >= 0 else -0.4),
             f"{d:+.2f}", ha="center", fontsize=8.5,
             fontweight="bold", color=bar_c[xi])

# Right: skip rate vs N
ax3 = axes[2]
ax3.plot(N_vals, df_sum["roll_skip_%"].tolist(),  "o-",
         color="#27AE60", linewidth=2, markersize=8, label="Rolling")
ax3.plot(N_vals, df_sum["calib_skip_%"].tolist(), "s-",
         color="#E67E22", linewidth=2, markersize=8, label="Calib a=0.01")
ax3.axhspan(15, 35, alpha=0.08, color="#27AE60", label="Target 15-35%")
ax3.set_xscale("log")
ax3.set_xlabel("N (log scale)", fontsize=11)
ax3.set_ylabel("Skip rate (%)", fontsize=11)
ax3.set_title("Skip rate vs instance size", fontsize=10, pad=6)
ax3.legend(fontsize=9, frameon=False)
ax3.grid(axis="y", alpha=0.2, linestyle="--")
for n, rs, cs in zip(N_vals, df_sum["roll_skip_%"], df_sum["calib_skip_%"]):
    ax3.annotate(f"N={n}", (n, max(rs, cs)), textcoords="offset points",
                 xytext=(0, 6), ha="center", fontsize=7.5)

plt.tight_layout()
p1 = RES_DIR / "benchmark_instances_plot.png"
fig.savefig(p1, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Plot -> {p1}")

# ── Plot 2: gain vs no-gate for all methods across instances ──────────────────
fig2, ax4 = plt.subplots(figsize=(13, 5))
fig2.suptitle(
    "Gain over no-gate by method and instance   seed=11  alpha=0.01  floor=0.90",
    fontsize=11, y=1.02,
)
offsets2 = [-1.5, -0.5, 0.5, 1.5]
for mi, method in enumerate(["fixed_gate","rolling","calibration"]):
    sub = df[df["method"] == method].sort_values("N")
    vals = sub["gain_vs_nogate"].tolist()
    ax4.bar(x + offsets2[mi+1]*width, vals, width*0.9,
            color=COLORS[method], alpha=0.8, label=LABELS[method], edgecolor="white")
ax4.axhline(0, color="#555", linewidth=1.2, linestyle="--")
ax4.set_xticks(x)
ax4.set_xticklabels([f"{c}\nN={n}" for c, n in zip(cities, N_vals)], fontsize=9)
ax4.set_ylabel("Gain vs no-gate", fontsize=11)
ax4.set_title("How much each method improves over no-gate", fontsize=10, pad=6)
ax4.legend(fontsize=10, frameon=False)
ax4.grid(axis="y", alpha=0.2, linestyle="--")
plt.tight_layout()
p2 = RES_DIR / "benchmark_gain_vs_nogate.png"
fig2.savefig(p2, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig2)

plt.rcdefaults()
print(f"Plot -> {p2}")
print("\nSummary:")
print(df_sum[["city","N","no_gate","rolling","calib","delta_calib_roll",
              "roll_skip_%","calib_skip_%"]].to_string(index=False))
