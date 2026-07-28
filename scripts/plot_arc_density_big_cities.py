"""
plot_arc_density_big_cities.py
================================
Histogram of arc density (# of other candidate sites within D=2km) across
all candidate sites, for the 4 large city instances (Napoli, Milano, Roma,
Torino) used throughout the reconstruction benchmarks.

Usage:
    python scripts/plot_arc_density_big_cities.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evcs.geom import build_arcs

DATA_DIR = PROJECT_ROOT / "data" / "input"
OUT_DIR  = PROJECT_ROOT / "results" / "diagnostics" / "demand_clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D_COVER = 2.0  # km, same coverage radius used everywhere else in the pipeline

CITIES = {
    "Napoli": ("center_547_Napoli_k3525.csv", "#C0392B"),
    "Milano": ("center_87_Milano_k3375.csv",  "#2471A3"),
    "Roma":   ("center_432_Roma_k4400.csv",   "#27AE60"),
    "Torino": ("center_207_Torino_k1950.csv", "#E67E22"),
}

degree_data = {}
for city, (csv_name, color) in CITIES.items():
    df        = pd.read_csv(DATA_DIR / csv_name)
    coords_km = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float) * 111.0
    M         = coords_km.shape[0]

    _, in_range, _, _ = build_arcs(coords_km, coords_km, D=D_COVER, forbid_self=False)
    degree = np.zeros(M)
    for (i, j) in in_range:
        degree[j] += 1

    degree_data[city] = dict(degree=degree, color=color, M=M)
    print(f"{city:<8} N={M:<5}  mean={degree.mean():7.1f}  median={np.median(degree):7.1f}  "
          f"min={degree.min():.0f}  max={degree.max():.0f}  std={degree.std():.1f}")

# ── Plot ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
axes = axes.flatten()
bins = np.linspace(0, max(d["degree"].max() for d in degree_data.values()), 40)

for ax, (city, d) in zip(axes, degree_data.items()):
    ax.hist(d["degree"], bins=bins, density=True, histtype="stepfilled",
            color=d["color"], alpha=0.55, edgecolor=d["color"], linewidth=1.4)
    ax.axvline(d["degree"].mean(), color="black", linewidth=1.2, linestyle=":")
    ax.set_title(f"{city}  (N={d['M']}, mean={d['degree'].mean():.0f}, "
                 f"std={d['degree'].std():.0f})", fontsize=10.5)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

for ax in axes[2:]:
    ax.set_xlabel(f"Arc density  (# sites within {D_COVER} km)", fontsize=10)
for ax in axes[::2]:
    ax.set_ylabel("Density", fontsize=10)

fig.suptitle("Arc density distribution — Napoli, Milano, Roma, Torino", fontsize=13, y=1.0)
plt.tight_layout()
plot_path = OUT_DIR / "arc_density_big_cities.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
plt.rcdefaults()
print(f"\nPlot -> {plot_path}")
