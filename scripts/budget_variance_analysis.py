"""
budget_variance_analysis.py
============================
Compares within-city seed-to-seed variance of the heap_cache-minus-heap_only
delta at the original 120s budget vs. a longer budget (300s), for Napoli,
to answer: does more time per seed shrink the noise, or is the noise a
property of the effect itself (in which case a full budget-slicing
experiment across all 5 cities isn't worth running)?

Reads:
  results/diagnostics/gini_cache_effect/multiseed_reliability_per_seed.csv  (120s baseline, all cities)
  results/diagnostics/gini_cache_effect/budget_variance_per_run.csv         (300s Napoli re-run)

Usage:
    python scripts/budget_variance_analysis.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GCE_DIR = PROJECT_ROOT / "results" / "diagnostics" / "gini_cache_effect"
OUT_DIR = GCE_DIR

CITY = "Napoli"


def main():
    baseline = pd.read_csv(GCE_DIR / "multiseed_reliability_per_seed.csv")
    baseline = baseline[baseline["city"] == CITY][["seed", "delta"]].copy()
    baseline["budget_s"] = 120.0

    longrun = pd.read_csv(GCE_DIR / "budget_variance_per_run.csv")
    longrun = longrun[longrun["city"] == CITY]
    piv = longrun.pivot_table(index=["seed", "budget_s"], columns="label", values="best_score").reset_index()
    piv["delta"] = piv["heap_cache"] - piv["heap_only"]
    piv = piv[["seed", "budget_s", "delta"]]

    both = pd.concat([baseline, piv], ignore_index=True).sort_values(["budget_s", "seed"])
    per_seed_csv = OUT_DIR / "budget_variance_comparison_per_seed.csv"
    both.to_csv(per_seed_csv, index=False)
    print(f"CSV -> {per_seed_csv}\n")
    print(both.to_string(index=False))

    summary = (
        both.groupby("budget_s")["delta"]
            .agg(n_seeds="count", mean_delta="mean", std_delta="std",
                 min_delta="min", max_delta="max")
            .reset_index()
    )
    summary["cv"] = (summary["std_delta"] / summary["mean_delta"].abs()).round(3)
    summary["sign_flip"] = summary.apply(
        lambda r: (r["min_delta"] < 0) and (r["max_delta"] > 0), axis=1
    )
    summary_csv = OUT_DIR / "budget_variance_comparison_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"\nCSV -> {summary_csv}\n")
    print(summary.to_string(index=False))

    b120 = summary[summary["budget_s"] == 120.0].iloc[0]
    b300 = summary[summary["budget_s"] == 300.0].iloc[0] if (summary["budget_s"] == 300.0).any() else None

    if b300 is not None:
        std_ratio = b300["std_delta"] / b120["std_delta"]
        print(f"\nstd at 120s = {b120['std_delta']:.2f}   std at 300s = {b300['std_delta']:.2f}"
              f"   ratio (300s/120s) = {std_ratio:.2f}")
        if std_ratio < 0.6:
            verdict = ("VARIANCE SHRINKS meaningfully with budget (ratio<0.6) — the noise looks "
                       "like search-convergence noise, not an intrinsic property of the effect. "
                       "Worth running the full budget-slicing experiment across all 5 cities.")
        elif std_ratio > 0.85:
            verdict = ("VARIANCE DOES NOT SHRINK (ratio>0.85) even at 2.5x the budget — the "
                       "instability looks intrinsic to the effect (demand-realization / destroy-repair "
                       "randomness dominates regardless of search time), not a convergence artifact. "
                       "A full budget-slicing experiment is unlikely to rescue the predictor; honest "
                       "conclusion: no reliable structural predictor found for the heap+cache effect.")
        else:
            verdict = ("PARTIAL reduction (ratio 0.6-0.85) — budget helps some but seed noise still "
                       "dominates. Marginal case; more seeds would help more than more budget.")
        print(f"\n{verdict}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, color="#888", linewidth=1.2, linestyle="--", zorder=1)
    colors = {120.0: "#C0392B", 300.0: "#2471A3"}
    for budget, c in colors.items():
        sub = both[both["budget_s"] == budget]
        if sub.empty:
            continue
        jitter = np.random.default_rng(0).uniform(-0.08, 0.08, len(sub))
        ax.scatter(np.full(len(sub), budget) + jitter * 40, sub["delta"], color=c, alpha=0.6,
                   s=70, zorder=3, edgecolors="white", linewidths=0.8, label=f"{int(budget)}s (n={len(sub)})")
        mean_d = sub["delta"].mean()
        std_d = sub["delta"].std()
        ax.errorbar([budget], [mean_d], yerr=[std_d], fmt="D", markersize=14, color=c,
                    ecolor=c, capsize=6, elinewidth=1.8, markeredgecolor="black",
                    markeredgewidth=1.2, zorder=4)
    ax.set_xlim(80, 340)
    ax.set_xlabel("Per-(config,seed) time budget (s)", fontsize=11)
    ax.set_ylabel("heap_cache $-$ heap_only best score", fontsize=11)
    ax.set_title(f"{CITY}: does a longer budget shrink seed-to-seed variance?", fontsize=12)
    ax.legend(fontsize=9.5, frameon=False)
    ax.grid(alpha=0.2, linestyle="--")
    plt.tight_layout()
    plot_path = OUT_DIR / "budget_variance_comparison.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcdefaults()
    print(f"\nPlot -> {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
