"""
arc_density_bimodality_predictor.py
=====================================
Follow-up to gini_cache_effect_combined.py: tests whether mean arc density,
or a bimodality score (Hartigan's dip statistic) of the arc-density
distribution, predicts the sign/magnitude of the distance-weighted partial
cache effect (heap_cache - heap_only) better than Gini did (Gini was
essentially null once size was controlled for: r=-0.395 on the same-size
trio, r=0.235 pooled — see project_gini_cache_effect_hypothesis memory).

Cache delta values are the mean-across-seeds heap_cache-minus-heap_only
scores already collected:
  Verona=0 (ceiling), Monza=0 (ceiling), Napoli=+13.4 (n=3), Roma=-16.5 (n=3),
  Milano=+11.2 (n=7), Torino=+20.8 (n=7)
pulled directly from results/diagnostics/gini_cache_effect/
gini_cache_effect_combined_summary.csv so the exact (unrounded) values are used.

Verona and Monza are "ceiling" cases (the reconstruction fully converges to
the same score regardless of config within the 120s budget — see
project_gini_cache_effect_hypothesis memory) and are flagged with a distinct
marker rather than treated as genuine zero-effect data points.

Usage:
    python scripts/arc_density_bimodality_predictor.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import diptest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evcs.geom import build_arcs

DATA_DIR   = PROJECT_ROOT / "data" / "input"
OUT_DIR    = PROJECT_ROOT / "results" / "diagnostics" / "gini_cache_effect"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "gini_cache_effect_combined_summary.csv"

D_COVER = 2.0

CITY_CSV = {
    "Verona":  "center_146_Verona_k250.csv",
    "Monza":   "center_79_Monza_k400.csv",
    "Napoli":  "center_547_Napoli_k3525.csv",
    "Milano":  "center_87_Milano_k3375.csv",
    "Roma":    "center_432_Roma_k4400.csv",
    "Torino":  "center_207_Torino_k1950.csv",
    "Genova":  "center_276_Genova_k825.csv",
}
CEILING_CITIES = {"Verona", "Monza"}
COLOR = {
    "Verona": "#8E44AD", "Monza": "#16A085", "Napoli": "#C0392B",
    "Roma": "#27AE60", "Milano": "#2471A3", "Torino": "#E67E22",
    "Genova": "#F1C40F",
}
# extension runs use --out-tag <suffix>, writing gini_cache_effect_per_run_<suffix>.csv
# genova60 (budget shrunk to 60s to avoid the ceiling effect seen at 120s) takes
# precedence over the original 120s genova run, which hit the ceiling (delta=0).
EXTRA_PER_RUN_TAGS = ["genova60", "genova"]


def load_delta_for(city):
    combined = pd.read_csv(SUMMARY_CSV).set_index("city")["mean_delta"]
    if city in combined.index:
        return float(combined[city])
    for tag in EXTRA_PER_RUN_TAGS:
        path = OUT_DIR / f"gini_cache_effect_per_run_{tag}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["city"] == city]
        if df.empty:
            continue
        piv = df.pivot_table(index="seed", columns="label", values="best_score")
        delta = (piv["heap_cache"] - piv["heap_only"]).mean()
        return float(delta)
    raise KeyError(f"No cache_delta found for {city}; run gini_cache_effect_experiment.py --city {city} --out-tag <tag> first")


def main():
    delta_df = {city: load_delta_for(city) for city in CITY_CSV}

    rows = []
    for city, csv_name in CITY_CSV.items():
        df        = pd.read_csv(DATA_DIR / csv_name)
        coords_km = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float) * 111.0
        M         = coords_km.shape[0]

        _, in_range, _, _ = build_arcs(coords_km, coords_km, D=D_COVER, forbid_self=False)
        degree = np.zeros(M)
        for (i, j) in in_range:
            degree[j] += 1

        dip, pval = diptest.diptest(degree)
        rows.append({
            "city": city, "N": M,
            "mean_degree": float(degree.mean()),
            "std_degree": float(degree.std()),
            "dip_stat": float(dip),
            "dip_pval": float(pval),
            "cache_delta": float(delta_df[city]),
            "is_ceiling": city in CEILING_CITIES,
        })
        print(f"{city:<8} N={M:<5}  mean_degree={degree.mean():7.1f}  "
              f"dip={dip:.4f}  dip_p={pval:.4f}  cache_delta={delta_df[city]:+7.2f}")

    res = pd.DataFrame(rows)
    out_csv = OUT_DIR / "arc_density_bimodality_predictors.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nCSV -> {out_csv}\n")
    print(res[["city", "N", "mean_degree", "dip_stat", "dip_pval", "cache_delta", "is_ceiling"]]
          .to_string(index=False))

    # ── Correlations ────────────────────────────────────────────────────────────
    r_deg_all  = res["mean_degree"].corr(res["cache_delta"])
    r_dip_all  = res["dip_stat"].corr(res["cache_delta"])
    non_ceil   = res[~res["is_ceiling"]]
    r_deg_noc  = non_ceil["mean_degree"].corr(non_ceil["cache_delta"])
    r_dip_noc  = non_ceil["dip_stat"].corr(non_ceil["cache_delta"])

    n_all, n_noc = len(res), len(non_ceil)
    print(f"\nPearson r (mean arc density vs delta), all {n_all} cities        = {r_deg_all:.3f}")
    print(f"Pearson r (mean arc density vs delta), {n_noc} non-ceiling      = {r_deg_noc:.3f}")
    print(f"Pearson r (dip statistic   vs delta), all {n_all} cities        = {r_dip_all:.3f}")
    print(f"Pearson r (dip statistic   vs delta), {n_noc} non-ceiling      = {r_dip_noc:.3f}")

    # ── Plot ────────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    specs = [
        (axes[0], "mean_degree", "Mean arc density  (# sites within 2.0 km)", r_deg_all),
        (axes[1], "dip_stat",    "Hartigan dip statistic (bimodality score)", r_dip_all),
    ]
    for ax, col, xlabel, r_all in specs:
        ax.axhline(0, color="#888", linewidth=1.2, linestyle="--", zorder=1)
        for _, row in res.iterrows():
            marker = "X" if row["is_ceiling"] else "o"
            size = 130 if row["is_ceiling"] else 150
            ax.scatter(row[col], row["cache_delta"], marker=marker, s=size,
                       color=COLOR[row["city"]], edgecolors="white", linewidths=1.0,
                       zorder=3)
            label = f"{row['city']} (ceiling)" if row["is_ceiling"] else row["city"]
            ax.annotate(label, (row[col], row["cache_delta"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=9.5)

        z = np.polyfit(res[col], res["cache_delta"], 1)
        xs = np.linspace(res[col].min(), res[col].max(), 50)
        pad = 0.06 * (res[col].max() - res[col].min())
        xs = np.linspace(res[col].min() - pad, res[col].max() + pad, 50)
        ax.plot(xs, np.polyval(z, xs), color="#555", linewidth=1.3, linestyle=":",
                label=f"linear fit, all {len(res)} (r={r_all:.2f})")

        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.set_ylabel("heap_cache score $-$ heap_only score", fontsize=10.5)
        ax.legend(fontsize=9, frameon=False, loc="best")
        ax.grid(alpha=0.2, linestyle="--")

    axes[0].set_title("Mean arc density vs. cache effect", fontsize=11.5)
    axes[1].set_title("Bimodality (dip stat) vs. cache effect", fontsize=11.5)
    fig.suptitle("Does network structure predict the cache effect better than Gini?",
                 fontsize=13, y=1.02)

    plt.tight_layout()
    plot_path = OUT_DIR / "arc_density_bimodality_vs_cache_effect.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcdefaults()
    print(f"\nPlot -> {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
