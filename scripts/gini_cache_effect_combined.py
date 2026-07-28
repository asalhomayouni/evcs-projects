"""
gini_cache_effect_combined.py
==============================
Combines all data collected so far for the Gini-vs-cache-effect hypothesis
and produces an honest final read, addressing the two problems found in the
first pass (gini_cache_effect_experiment.py):

  1. Verona/Monza showed exactly-zero delta because they're small enough to
     fully converge within the 120s budget regardless of config (a ceiling
     effect confounded with their low Gini, not real evidence either way).
  2. Milano/Torino's positive mean delta had std > mean at only 3 seeds
     (one seed even flipped sign for both cities) — not statistically robust.

This script:
  (a) adds a clean same-size-class comparison: Napoli (N=3525), Milano
      (N=3375), Roma (N=4400) span Gini 0.259 -> 0.317 -> 0.330 at nearly
      the same instance size, so size isn't a confound for this trio. Napoli
      and Roma data are reused as-is from controlled_reconstruction_experiment
      (same P_t=117, same seeds, same code path) rather than re-run.
  (b) extends Milano and Torino from 3 to 7 seeds (11/22/33 + 44/55/66/77,
      run separately with --out-tag extra) to see if the effect survives a
      larger sample.

Reads from:
  results/diagnostics/controlled_reconstruction/controlled_per_run.csv   (Napoli, Roma)
  results/diagnostics/gini_cache_effect/gini_cache_effect_per_run.csv     (Verona, Monza, Milano, Torino: seeds 11/22/33)
  results/diagnostics/gini_cache_effect/gini_cache_effect_per_run_extra.csv (Milano, Torino: seeds 44/55/66/77)

Usage:
    python scripts/gini_cache_effect_combined.py
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CR_DIR   = PROJECT_ROOT / "results" / "diagnostics" / "controlled_reconstruction"
GCE_DIR  = PROJECT_ROOT / "results" / "diagnostics" / "gini_cache_effect"
GINI_CSV = PROJECT_ROOT / "results" / "diagnostics" / "demand_clustering" / "gini_all_cities.csv"
OUT_DIR  = GCE_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAME_SIZE_TRIO = {"Napoli", "Milano", "Roma"}  # N = 3525 / 3375 / 4400


def main():
    gini_df = pd.read_csv(GINI_CSV).set_index("city")["gini_coverage"]

    cr = pd.read_csv(CR_DIR / "controlled_per_run.csv")
    cr = cr[cr["city"].isin(["Napoli", "Roma"]) & cr["label"].isin(["heap_only", "heap_cache"])]
    cr = cr[["city", "label", "seed", "best_score", "N"]]

    gce = pd.read_csv(GCE_DIR / "gini_cache_effect_per_run.csv")
    gce = gce[["city", "label", "seed", "best_score", "N"]]

    extra_path = GCE_DIR / "gini_cache_effect_per_run_extra.csv"
    frames = [cr, gce]
    if extra_path.exists():
        extra = pd.read_csv(extra_path)[["city", "label", "seed", "best_score", "N"]]
        frames.append(extra)
    else:
        print("NOTE: extra-seeds file not found yet; using only the original 3 seeds for Milano/Torino.")

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["city", "label", "seed"])
    df["gini_coverage"] = df["city"].map(gini_df)

    # ── Per-seed paired delta ──────────────────────────────────────────────────
    piv = df.pivot_table(index=["city", "seed"], columns="label", values="best_score").reset_index()
    piv["delta"] = piv["heap_cache"] - piv["heap_only"]
    piv["gini_coverage"] = piv["city"].map(gini_df)
    piv["N"] = piv["city"].map(df.groupby("city")["N"].first())

    per_seed_csv = OUT_DIR / "gini_cache_effect_combined_per_seed.csv"
    piv.to_csv(per_seed_csv, index=False)

    summary = (
        piv.groupby("city")
           .agg(gini_coverage=("gini_coverage", "first"), N=("N", "first"),
                n_seeds=("delta", "count"),
                mean_delta=("delta", "mean"), std_delta=("delta", "std"),
                min_delta=("delta", "min"), max_delta=("delta", "max"))
           .reset_index()
           .sort_values("gini_coverage")
    )
    summary["same_size_trio"] = summary["city"].isin(SAME_SIZE_TRIO)
    summary["se_delta"] = summary["std_delta"] / np.sqrt(summary["n_seeds"].clip(lower=1))
    summary["t_stat"] = summary["mean_delta"] / summary["se_delta"].replace(0, np.nan)

    summary_csv = OUT_DIR / "gini_cache_effect_combined_summary.csv"
    summary.to_csv(summary_csv, index=False)

    print(f"CSV -> {per_seed_csv}")
    print(f"CSV -> {summary_csv}\n")
    print(summary[["city", "N", "gini_coverage", "n_seeds", "mean_delta", "std_delta",
                    "t_stat", "same_size_trio"]].to_string(index=False))

    # ── Same-size trio: the clean test ─────────────────────────────────────────
    trio = summary[summary["same_size_trio"]].sort_values("gini_coverage")
    print(f"\n{'─'*70}")
    print("  SAME-SIZE TRIO (Napoli N=3525, Milano N=3375, Roma N=4400)")
    print(f"{'─'*70}")
    print(trio[["city", "N", "gini_coverage", "mean_delta", "std_delta", "t_stat"]].to_string(index=False))
    trio_corr = trio["gini_coverage"].corr(trio["mean_delta"])
    print(f"  Pearson r (size-controlled) = {trio_corr:.3f}")

    # ── All cities, all available seeds ────────────────────────────────────────
    all_corr = summary["gini_coverage"].corr(summary["mean_delta"])
    print(f"\n  Pearson r (all {len(summary)} cities, pooled seeds) = {all_corr:.3f}")

    # ── Plot ────────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: all cities, marker shape encodes same-size-trio membership
    ax = axes[0]
    ax.axhline(0, color="#888", linewidth=1.2, linestyle="--", zorder=1)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(summary)))
    for (_, row), c in zip(summary.iterrows(), colors):
        marker = "D" if row["same_size_trio"] else "o"
        ax.errorbar(row["gini_coverage"], row["mean_delta"], yerr=row["se_delta"],
                    fmt=marker, markersize=12, color=c, ecolor=c, capsize=4,
                    elinewidth=1.2, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        ax.annotate(f"{row['city']}\n(N={int(row['N'])}, n={int(row['n_seeds'])})",
                    (row["gini_coverage"], row["mean_delta"]),
                    textcoords="offset points", xytext=(9, 4), fontsize=8.5)
    ax.set_xlabel("Gini coefficient of reachable population per site", fontsize=10.5)
    ax.set_ylabel("heap_cache score $-$ heap_only score  (mean $\\pm$ SE)", fontsize=10.5)
    ax.set_title("All cities  (diamonds = same-size trio N=3375-4400)", fontsize=11)
    ax.grid(alpha=0.2, linestyle="--")

    # Right: same-size trio only, zoomed
    axR = axes[1]
    axR.axhline(0, color="#888", linewidth=1.2, linestyle="--", zorder=1)
    for (_, row), c in zip(trio.iterrows(), plt.cm.viridis(np.linspace(0.15, 0.85, len(trio)))):
        axR.errorbar(row["gini_coverage"], row["mean_delta"], yerr=row["se_delta"],
                     fmt="D", markersize=14, color=c, ecolor=c, capsize=5,
                     elinewidth=1.4, markeredgecolor="white", markeredgewidth=1.0, zorder=3)
        axR.annotate(f"{row['city']} (n={int(row['n_seeds'])})",
                     (row["gini_coverage"], row["mean_delta"]),
                     textcoords="offset points", xytext=(9, 4), fontsize=9.5)
    if len(trio) >= 2:
        z = np.polyfit(trio["gini_coverage"], trio["mean_delta"], 1)
        xs = np.linspace(trio["gini_coverage"].min() - 0.01, trio["gini_coverage"].max() + 0.01, 50)
        axR.plot(xs, np.polyval(z, xs), color="#555", linewidth=1.3, linestyle=":",
                 label=f"linear fit (r={trio_corr:.2f})")
        axR.legend(fontsize=9, frameon=False)
    axR.set_xlabel("Gini coefficient of reachable population per site", fontsize=10.5)
    axR.set_ylabel("heap_cache score $-$ heap_only score  (mean $\\pm$ SE)", fontsize=10.5)
    axR.set_title("Same-size trio only (size-controlled test)", fontsize=11)
    axR.grid(alpha=0.2, linestyle="--")

    fig.suptitle("Does site-value inequality predict the cache effect?", fontsize=13, y=1.02)
    plt.tight_layout()
    plot_path = OUT_DIR / "gini_vs_cache_effect_combined.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcdefaults()
    print(f"\nPlot -> {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
