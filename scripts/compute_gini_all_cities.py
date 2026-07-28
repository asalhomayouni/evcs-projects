"""
compute_gini_all_cities.py
===========================
Step 1 of the Gini-vs-cache-effect hypothesis test: compute the Gini
coefficient of reachable population per candidate site (D=2.0 km arcs, same
definition as the reconstruction) for every city instance we have, so we can
pick low- and high-Gini cities for the controlled reconstruction experiment.

Usage:
    python scripts/compute_gini_all_cities.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evcs.geom import build_arcs

DATA_DIR = PROJECT_ROOT / "data" / "input"
OUT_DIR  = PROJECT_ROOT / "results" / "diagnostics" / "demand_clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D_COVER = 2.0  # km, same coverage radius used everywhere else in the pipeline

CITIES = {
    "Vicenza": "center_102_Vicenza_k125.csv",
    "Verona":  "center_146_Verona_k250.csv",
    "Monza":   "center_79_Monza_k400.csv",
    "Genova":  "center_276_Genova_k825.csv",
    "Palermo": "center_710_Palermo_k1025.csv",
    "Torino":  "center_207_Torino_k1950.csv",
    "Milano":  "center_87_Milano_k3375.csv",
    "Napoli":  "center_547_Napoli_k3525.csv",
    "Roma":    "center_432_Roma_k4400.csv",
}


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


rows = []
for city, csv_name in CITIES.items():
    df        = pd.read_csv(DATA_DIR / csv_name)
    coords_km = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float) * 111.0
    pop       = np.maximum(df["Aggregated_Population"].to_numpy(float), 0.0)
    M         = coords_km.shape[0]

    distIJ, in_range, Ji, Ij = build_arcs(coords_km, coords_km, D=D_COVER, forbid_self=False)

    degree   = np.zeros(M)
    coverage = np.zeros(M)
    for (i, j) in in_range:
        degree[j] += 1
        coverage[j] += pop[i]

    g_deg = gini(degree)
    g_cov = gini(coverage)
    rows.append({
        "city": city, "N": M,
        "mean_degree": round(float(degree.mean()), 2),
        "gini_degree": round(g_deg, 4),
        "gini_coverage": round(g_cov, 4),
    })
    print(f"{city:<9} N={M:<5}  mean_degree={degree.mean():7.1f}  "
          f"gini(degree)={g_deg:.3f}  gini(coverage)={g_cov:.3f}")

df_out = pd.DataFrame(rows).sort_values("gini_coverage").reset_index(drop=True)
out_csv = OUT_DIR / "gini_all_cities.csv"
df_out.to_csv(out_csv, index=False)
print(f"\nSorted by gini_coverage (ascending = most equal/uniform first):")
print(df_out.to_string(index=False))
print(f"\nCSV -> {out_csv}")
