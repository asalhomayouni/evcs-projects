"""
plot_ls_dr_responsibility.py — stacked bar chart of new-incumbent responsibility
(local search vs. raw destroy-reconstruct) under the no-gate configuration,
across twelve Italian city instances.

Input CSV columns: city, local_search_pct, destroy_reconstruct_pct
(the two must sum to 100 per row).

Usage:
    python scripts/plot_ls_dr_responsibility.py
    python scripts/plot_ls_dr_responsibility.py --csv path/to/data.csv --out ls_dr_responsibility.png
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "results" / "gate" / "ls_dr_no_gate.csv"

COLOR_LS = "#2a78d6"   # local search
COLOR_DR = "#eb6834"   # destroy-reconstruct
GRIDLINE = "#e1e0d9"
INK      = "#0b0b0b"
MUTED    = "#52514e"

p = argparse.ArgumentParser()
p.add_argument("--csv", default=str(DEFAULT_CSV))
p.add_argument("--out", default="ls_dr_responsibility.png")
args = p.parse_args()

df = pd.read_csv(args.csv)
df = df.sort_values("local_search_pct", ascending=False).reset_index(drop=True)

cities = df["city"].tolist()
ls_pct = df["local_search_pct"].to_numpy()
dr_pct = df["destroy_reconstruct_pct"].to_numpy()

y = range(len(cities))

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

bar_h = 0.62
ax.barh(y, ls_pct, height=bar_h, color=COLOR_LS, label="Local search",
        edgecolor="white", linewidth=0.6, zorder=3)
ax.barh(y, dr_pct, height=bar_h, left=ls_pct, color=COLOR_DR, label="Destroy-reconstruct",
        edgecolor="white", linewidth=0.6, zorder=3)

# selective direct labels — only in segments wide enough to hold text cleanly
for yi, (ls, dr) in enumerate(zip(ls_pct, dr_pct)):
    if ls >= 8:
        ax.text(ls / 2, yi, f"{ls:.0f}%", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold", zorder=4)
    if dr >= 8:
        ax.text(ls + dr / 2, yi, f"{dr:.0f}%", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold", zorder=4)

ax.set_yticks(list(y))
ax.set_yticklabels(cities, fontsize=10, color=INK)
ax.invert_yaxis()  # highest local_search_pct at top

ax.set_xlim(0, 100)
ax.set_xlabel("Share of new incumbents (%)", fontsize=11, color=INK)
ax.set_title(
    "Fraction of new incumbents found by local search vs.\ndestroy-reconstruct, across twelve Italian cities",
    fontsize=13, color=INK, pad=14,
)

ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

for spine in ("top", "right", "left"):
    ax.spines[spine].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="both", length=0, colors=MUTED)

ax.legend(
    loc="lower right", frameon=False, fontsize=10,
    labelcolor=INK, handlelength=1.2, handleheight=1.2,
)

plt.tight_layout()
fig.savefig(args.out, dpi=200, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {Path(args.out).resolve()}")
