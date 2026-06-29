"""
compose_monza_cp_vs_ue.py

Composes the existing Closest-Priority and User-Equilibrium individual PNGs
from results/diagnostics/policy_comparison/ into a clean side-by-side figure.

Output:
  results/diagnostics/policy_comparison/monza_cp_vs_ue_composed.pdf
  results/diagnostics/policy_comparison/monza_cp_vs_ue_composed.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IN_DIR  = PROJECT_ROOT / "results" / "diagnostics" / "policy_comparison"
OUT_DIR = IN_DIR

img_cp = mpimg.imread(str(IN_DIR / "closest_priority_center_79_Monza_k400.png"))
img_ue = mpimg.imread(str(IN_DIR / "ue_center_79_Monza_k400.png"))

plt.rcParams.update({"font.family": "serif"})

fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                         gridspec_kw={"wspace": 0.04})

for ax, img, label in zip(axes,
                           [img_cp, img_ue],
                           ["(a) Closest-Priority", "(b) User Equilibrium"]):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(label, fontsize=12, fontweight="bold", pad=8)

fig.suptitle("Monza — station placement comparison",
             fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout(pad=0.5)

out_pdf = OUT_DIR / "monza_cp_vs_ue_composed.pdf"
out_png = OUT_DIR / "monza_cp_vs_ue_composed.png"
fig.savefig(str(out_pdf), dpi=200, bbox_inches="tight")
fig.savefig(str(out_png), dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
print("Done.")
