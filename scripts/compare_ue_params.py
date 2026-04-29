"""
Side-by-side comparison of two ALNS-UE best solutions
(alpha_wait=1.0  vs  alpha_wait=10.0)  — Verona instance.

Loads the already-saved best_solution.png from each run directory
and stitches them into a single figure with annotations.

Usage:
    python scripts/compare_ue_params.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

SWEEP_DIR = PROJECT_ROOT / "results" / "diagnostics" / "param_sweep" / "center_146_Verona_k250"
OUT_DIR   = PROJECT_ROOT / "results" / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Locate the two runs (noopt=10, pd=5) ─────────────────────────────────────
runs = [
    {
        "label":       r"$\alpha_{wait}$ = 1.0",
        "sublabel":    "noopt=10  pd=5",
        "img_path":    SWEEP_DIR / "g1_aw1p0_noopt10p0.png",
        "ue_score":    189.636,
        "pct_served":  75.8,
        "n_stations":  10,
        "n_chargers":  48,
        "n_iter_best": 0,
        "note":        "noopt=10  pd=5\nimprov=0",
        "color":       "#2166ac",
    },
    {
        "label":       r"$\alpha_{wait}$ = 10.0",
        "sublabel":    "noopt=10  pd=5",
        "img_path":    SWEEP_DIR / "g1_aw10p0_noopt10p0.png",
        "ue_score":    175.514,
        "pct_served":  70.1,
        "n_stations":  10,
        "n_chargers":  48,
        "n_iter_best": 0,
        "note":        "noopt=10  pd=5\nimprov=0",
        "color":       "#d6604d",
    },
]

# ── Load images ───────────────────────────────────────────────────────────────
imgs = []
for r in runs:
    p = r["img_path"]
    if not p.exists():
        raise FileNotFoundError(f"Missing: {p}")
    imgs.append(mpimg.imread(str(p)))

# ── Build figure ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 11))
fig.patch.set_facecolor("#f8f8f8")

# Super title
fig.text(
    0.5, 0.97,
    "UE parameter comparison — Verona (N=250)  |  "
    r"Effect of $\alpha_{wait}$ on station placement  (noopt=10, pd=5)",
    ha="center", va="top", fontsize=15, fontweight="bold", color="#222222",
)
fig.text(
    0.5, 0.935,
    "Same instance · 20 D-R iterations · same site capacity · penalty_dist=5",
    ha="center", va="top", fontsize=11, color="#555555",
)

# Two image panels (left / right)
ax_imgs = [
    fig.add_axes([0.02,  0.08, 0.46, 0.83]),
    fig.add_axes([0.52,  0.08, 0.46, 0.83]),
]

for ax, img, r in zip(ax_imgs, imgs, runs):
    ax.imshow(img, aspect="auto")
    ax.axis("off")

    # Colored border frame
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(r["color"])
        spine.set_linewidth(4)

    # Title bar above image (within axes, top)
    ax.text(
        0.5, 1.01,
        f"{r['label']}   ({r['sublabel']})",
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=13, fontweight="bold", color=r["color"],
    )

    # Metric box — bottom strip
    metrics = (
        f"UE score: {r['ue_score']:.3f}   "
        f"served: {r['pct_served']:.1f}%   "
        f"stations: {r['n_stations']}   "
        f"chargers: {r['n_chargers']}   "
        f"first improved at iter: {r['n_iter_best']}"
    )
    ax.text(
        0.5, -0.01, metrics,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=9.5, family="monospace", color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=r["color"], linewidth=1.5, alpha=0.9),
    )

    # Note badge — top-left corner of image
    ax.text(
        0.02, 0.98, r["note"],
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8.5, color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=r["color"], alpha=0.85),
    )

# ── Delta annotation in the middle gap ───────────────────────────────────────
ax_mid = fig.add_axes([0.465, 0.35, 0.07, 0.30])
ax_mid.axis("off")

delta_ue  = runs[0]["ue_score"] - runs[1]["ue_score"]
delta_pct = runs[0]["pct_served"] - runs[1]["pct_served"]

ax_mid.text(
    0.5, 0.82, "vs", ha="center", va="center",
    fontsize=20, fontweight="bold", color="#888888",
    transform=ax_mid.transAxes,
)

delta_lines = [
    r"$\Delta$ UE score",
    f"+{delta_ue:.3f}",
    "",
    r"$\Delta$ served",
    f"+{delta_pct:.1f}pp",
    "",
    r"($\alpha_{wait}$=1.0",
    " is higher)",
]

ax_mid.text(
    0.5, 0.45, "\n".join(delta_lines),
    ha="center", va="center", fontsize=9,
    transform=ax_mid.transAxes, color="#333333",
    bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffbe6",
              edgecolor="#ccaa00", linewidth=1.5),
)

# ── Footer note ───────────────────────────────────────────────────────────────
fig.text(
    0.5, 0.005,
    r"noopt=10: lower no-option cost → UE penalises unserved demand less → $\alpha_{wait}$=1.0 still serves more but gap narrows vs noopt=30.",
    ha="center", va="bottom", fontsize=9, color="#666666", style="italic",
)

out_path = OUT_DIR / "compare_ue_alpha_wait_noopt10.png"
fig.savefig(str(out_path), dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
