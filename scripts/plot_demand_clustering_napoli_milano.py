"""
plot_demand_clustering_napoli_milano.py
========================================
Tommaso's question: does Napoli have more obvious placement locations
(trivial solutions) than Milano because of its demand structure?

Compares the two cities' k-means demand-node instances side by side:
  (A) node maps (lon/lat, sized/colored by population)
  (B) arc-density distribution — # of other nodes within D=2km of each node,
      built with the same evcs.geom.build_arcs used by the solver
  (C) Lorenz curve + Gini coefficient of per-node "coverage" (total population
      reachable within D km) — the same raw_delta the greedy reconstruction
      scores sites by. A more unequal (higher-Gini) curve means a few sites
      dominate the greedy score and are picked almost regardless of method —
      i.e. more "trivial"/obvious placement locations.

Usage:
    python scripts/plot_demand_clustering_napoli_milano.py
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
}


def load(csv_name):
    df        = pd.read_csv(DATA_DIR / csv_name)
    lonlat    = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float)
    coords_km = lonlat * 111.0
    pop       = np.maximum(df["Aggregated_Population"].to_numpy(float), 0.0)
    return lonlat, coords_km, pop


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def lorenz_curve(x):
    x = np.sort(np.asarray(x, float))
    cum = np.cumsum(x) / x.sum()
    return np.concatenate([[0.0], cum])


data = {}
for city, (csv_name, color) in CITIES.items():
    lonlat, coords_km, pop = load(csv_name)
    M = coords_km.shape[0]
    distIJ, in_range, Ji, Ij = build_arcs(coords_km, coords_km, D=D_COVER, forbid_self=False)

    degree = np.zeros(M)
    coverage = np.zeros(M)
    for (i, j) in in_range:
        degree[j] += 1
        coverage[j] += pop[i]

    data[city] = dict(
        lonlat=lonlat, pop=pop, M=M, color=color,
        degree=degree, coverage=coverage,
        gini_deg=gini(degree), gini_cov=gini(coverage),
    )
    print(f"{city:<8} N={M:<5}  mean_degree={degree.mean():7.1f}  "
          f"median_degree={np.median(degree):7.1f}  max_degree={degree.max():7.0f}  "
          f"gini(degree)={data[city]['gini_deg']:.3f}  gini(coverage)={data[city]['gini_cov']:.3f}")

# ── Plot ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig = plt.figure(figsize=(13, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0], hspace=0.32, wspace=0.28)

# Panel A/B: node maps
for col, city in enumerate(CITIES):
    ax = fig.add_subplot(gs[0, col])
    d = data[city]
    sizes = 4 + 40 * (d["pop"] / d["pop"].max())
    sc = ax.scatter(d["lonlat"][:, 0], d["lonlat"][:, 1], c=d["pop"],
                     s=sizes, cmap="viridis", alpha=0.75, linewidths=0,
                     norm=matplotlib.colors.LogNorm(vmin=max(d["pop"].min(), 1.0),
                                                     vmax=d["pop"].max()))
    ax.set_title(f"{city}  (N={d['M']} demand nodes)", fontsize=11)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.set_aspect("equal")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Population", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

fig.text(0.5, 0.985, "Demand node distribution — Napoli vs Milano", ha="center",
          fontsize=13, fontweight="bold")

# Panel C: arc-density (degree) histogram, both cities overlaid
axC = fig.add_subplot(gs[1, 0])
for city in CITIES:
    d = data[city]
    axC.hist(d["degree"], bins=40, density=True, histtype="stepfilled",
              alpha=0.35, color=d["color"], edgecolor=d["color"], linewidth=1.5,
              label=f"{city}  (mean={d['degree'].mean():.0f})")
axC.set_xlabel(f"Arc density per node  (# nodes within {D_COVER} km)", fontsize=10)
axC.set_ylabel("Density", fontsize=10)
axC.set_title("Arc density distribution", fontsize=11)
axC.legend(fontsize=9, frameon=False)
axC.grid(axis="y", alpha=0.2, linestyle="--")

# Panel D: Lorenz curve of per-node coverage + Gini
axD = fig.add_subplot(gs[1, 1])
axD.plot([0, 1], [0, 1], color="#888", linewidth=1.2, linestyle="--", label="Equality")
for city in CITIES:
    d = data[city]
    lc = lorenz_curve(d["coverage"])
    xs = np.linspace(0, 1, len(lc))
    axD.plot(xs, lc, color=d["color"], linewidth=2.2,
              label=f"{city}  (Gini={d['gini_cov']:.3f})")
axD.set_xlabel("Cumulative fraction of nodes (sorted by coverage)", fontsize=10)
axD.set_ylabel("Cumulative fraction of reachable population", fontsize=10)
axD.set_title("Lorenz curve — site coverage concentration", fontsize=11)
axD.legend(fontsize=9, frameon=False, loc="upper left")
axD.grid(alpha=0.2, linestyle="--")

plot_path = OUT_DIR / "napoli_vs_milano_demand_clustering.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
plt.rcdefaults()
print(f"\nPlot -> {plot_path}")

# ── Summary CSV ────────────────────────────────────────────────────────────
rows = []
for city, d in data.items():
    rows.append({
        "city": city, "N": d["M"],
        "mean_degree": round(float(d["degree"].mean()), 2),
        "median_degree": round(float(np.median(d["degree"])), 2),
        "max_degree": round(float(d["degree"].max()), 2),
        "gini_degree": round(d["gini_deg"], 4),
        "gini_coverage": round(d["gini_cov"], 4),
        "top1pct_coverage_share": round(
            float(np.sort(d["coverage"])[::-1][:max(1, d["M"] // 100)].sum() / d["coverage"].sum()), 4),
    })
df_summary = pd.DataFrame(rows)
summary_csv = OUT_DIR / "napoli_vs_milano_summary.csv"
df_summary.to_csv(summary_csv, index=False)
print(f"CSV  -> {summary_csv}\n")
print(df_summary.to_string(index=False))
