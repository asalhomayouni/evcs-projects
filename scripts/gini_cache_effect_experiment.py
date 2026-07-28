"""
gini_cache_effect_experiment.py
================================
Tests the hypothesis: the sign of the distance-weighted partial-cache effect
(heap_cache score - heap_only score) is predicted by the Gini coefficient of
reachable population per candidate site — high Gini (unequal site values,
e.g. Milano 0.330) should benefit from cache; low Gini (uniform site values,
e.g. Napoli/Verona) should not.

compute_gini_all_cities.py ranked all 9 city instances by gini(coverage).
Vicenza (N=125, lowest Gini) is excluded here: at the fixed P_t=117 budget it
would need up to 702 cumulative installs against a max capacity of
6*125=750 (93.6% utilization) — the site pool saturates almost immediately,
trivializing the heap/cache comparison. Monza (N=400, gini=0.233, only 29%
utilization) is used as the next-lowest-Gini city with a safe size instead.

Selected cities (2 low, 2 high Gini):
  Verona  gini=0.201  N=250
  Monza   gini=0.233  N=400
  Milano  gini=0.330  N=3375
  Torino  gini=0.337  N=1950

Uses the exact same reconstruction code path as controlled_reconstruction_
experiment.py (imports load_instance/run_config from it) — same fixed
P_t=117 (N_ref=3525), same 3 seeds (11/22/33), same 120s/config/seed budget.
Only heap_only and heap_cache are run (cache_only is not part of this
hypothesis test).

Usage:
    python scripts/gini_cache_effect_experiment.py
    python scripts/gini_cache_effect_experiment.py --city Verona Monza Milano Torino
"""
import sys, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

import controlled_reconstruction_experiment as cre

DATA_DIR = PROJECT_ROOT / "data" / "input"
OUT_DIR  = PROJECT_ROOT / "results" / "diagnostics" / "gini_cache_effect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GINI_CSV = PROJECT_ROOT / "results" / "diagnostics" / "demand_clustering" / "gini_all_cities.csv"

CITY_CSV = {
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

DEFAULT_CITIES = ["Verona", "Monza", "Milano", "Torino"]

# (label, refresh_R, use_heap, use_dist_weight)
CONFIGS = [
    ("heap_only",  1.00, True, False),
    ("heap_cache", 0.20, True, True),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", default=DEFAULT_CITIES, choices=list(CITY_CSV.keys()))
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--budget", type=float, default=120.0)
    ap.add_argument("--nref", type=int, default=3525)
    ap.add_argument("--out-tag", type=str, default="",
                     help="suffix for output filenames, so an extension run doesn't clobber the original")
    args = ap.parse_args()
    tag = f"_{args.out_tag}" if args.out_tag else ""

    gini_df = pd.read_csv(GINI_CSV).set_index("city")

    P_t = max(4, args.nref // 30)
    P_T = [P_t] * cre.T_PERIODS
    print(f"N_ref={args.nref}  ->  P_t={P_t} for ALL cities  |  seeds={args.seeds}  "
          f"budget={args.budget}s/config/seed")
    print(f"Cities: {args.city}")

    all_results = []
    for city in args.city:
        csv_name = CITY_CSV[city]
        g_cov = float(gini_df.loc[city, "gini_coverage"])
        print(f"\n{'='*70}")
        print(f"  {city}   gini(coverage)={g_cov:.3f}   budget={args.budget}s/config/seed")
        print(f"{'='*70}")

        for seed in args.seeds:
            print(f"  Loading instance (seed={seed})...", end=" ", flush=True)
            inst, Ij_int, ue_eval, ue_fn, M = cre.load_instance(csv_name, seed, P_T)
            print(f"done  N={M}")

            for label, R, use_heap, use_dw in CONFIGS:
                r = cre.run_config(label, R, use_heap, use_dw, inst, Ij_int, P_T,
                                    ue_fn, args.budget, seed)
                r["city"] = city
                r["N"] = M
                r["gini_coverage"] = g_cov
                all_results.append(r)
                print(f"    [seed={seed}] {label:<11} iters={r['iters']:>4}  "
                      f"best={r['best_score']:>9.3f}")

            ue_eval.dispose()

    df = pd.DataFrame(all_results)
    per_run_csv = OUT_DIR / f"gini_cache_effect_per_run{tag}.csv"
    df.to_csv(per_run_csv, index=False)
    print(f"\nCSV -> {per_run_csv}")

    # ── Summary: mean score per (city, config) + delta ────────────────────────
    summary = (
        df.groupby(["city", "gini_coverage", "label"])["best_score"]
          .agg(["mean", "std"])
          .unstack("label")
    )
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary["delta_cache_minus_heap"] = (
        summary["mean_heap_cache"] - summary["mean_heap_only"]
    )
    summary["cache_helps"] = summary["delta_cache_minus_heap"] > 0
    summary = summary.sort_values("gini_coverage").reset_index(drop=True)

    summary_csv = OUT_DIR / f"gini_cache_effect_summary{tag}.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"CSV -> {summary_csv}\n")
    print(summary[["city", "gini_coverage", "mean_heap_only", "mean_heap_cache",
                    "delta_cache_minus_heap", "cache_helps"]].to_string(index=False))

    # ── Trend check ────────────────────────────────────────────────────────────
    corr = summary["gini_coverage"].corr(summary["delta_cache_minus_heap"])
    print(f"\nPearson correlation(gini_coverage, delta) = {corr:.3f}")
    if corr > 0.3:
        verdict = "CONSISTENT with hypothesis (higher Gini -> cache helps more)"
    elif corr < -0.3:
        verdict = "CONTRADICTS hypothesis (higher Gini -> cache helps LESS)"
    else:
        verdict = "NO CLEAR TREND (weak/no correlation)"
    print(f"Verdict: {verdict}")

    # ── Plot: Gini vs delta ────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.axhline(0, color="#888", linewidth=1.2, linestyle="--", zorder=1)

    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(summary)))
    for (_, row), c in zip(summary.iterrows(), colors):
        ax.errorbar(
            row["gini_coverage"], row["delta_cache_minus_heap"],
            yerr=np.sqrt(row.get("std_heap_only", 0)**2 + row.get("std_heap_cache", 0)**2),
            fmt="o", markersize=11, color=c, ecolor=c, capsize=4, elinewidth=1.2,
            markeredgecolor="white", markeredgewidth=1.0, zorder=3,
        )
        ax.annotate(row["city"], (row["gini_coverage"], row["delta_cache_minus_heap"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=10)

    # trend line
    z = np.polyfit(summary["gini_coverage"], summary["delta_cache_minus_heap"], 1)
    xs = np.linspace(summary["gini_coverage"].min() - 0.01, summary["gini_coverage"].max() + 0.01, 50)
    ax.plot(xs, np.polyval(z, xs), color="#555", linewidth=1.3, linestyle=":",
            zorder=2, label=f"linear fit (r={corr:.2f})")

    ax.set_xlabel("Gini coefficient of reachable population per site", fontsize=11)
    ax.set_ylabel("heap_cache score $-$ heap_only score", fontsize=11)
    ax.set_title("Does site-value inequality predict the cache effect?", fontsize=12, pad=10)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.2, linestyle="--")

    plt.tight_layout()
    plot_path = OUT_DIR / f"gini_vs_cache_effect{tag}.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcdefaults()
    print(f"\nPlot -> {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
