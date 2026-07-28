"""
multi_seed_reliability_analysis.py
===================================
Robustness check for the arc-density-vs-cache-effect predictor (see
project_gini_cache_effect_hypothesis memory: mean arc density r=0.807 on
5 non-ceiling cities, using seed-averaged heap_cache-minus-heap_only score
deltas). That r=0.807 was computed from an UNEVEN number of seeds per city
(Roma/Napoli/Genova n=3, Milano/Torino n=7) — this script equalizes every
city at n=7 seeds (11/22/33/44/55/66/77) and asks two questions:

  1. Does the seed-averaged correlation survive once every city has the
     same (larger) seed count, and how much does it move under seed
     resampling (bootstrap over which seeds you happened to draw)?
  2. Does treating each seed as its own data point (pooled, n=35) instead
     of collapsing to a per-city mean change the picture?

Also reports per-city mean/std/CV of the delta across seeds, and flags any
city where the delta's sign flips across seeds (the single-seed result for
that city would have been a coin flip, not signal).

"Cache-effect delta" here is heap_cache best_score minus heap_only
best_score (solution-quality, not a wall-clock speedup) — same metric as
every prior script in this chain (controlled_reconstruction_experiment.py,
gini_cache_effect_experiment.py, arc_density_bimodality_predictor.py).

Arc density (mean # candidate sites within D=2km of each site) is a
structural property of the site coordinates alone — it does not depend on
seed at all (no RNG anywhere in build_arcs), so it is computed once per
city, not once per seed. That's asserted, not just assumed: the function
that computes it takes no seed/rng argument.

Reads:
  results/diagnostics/controlled_reconstruction/controlled_per_run.csv       (Roma, Napoli: seeds 11/22/33)
  results/diagnostics/gini_cache_effect/gini_cache_effect_per_run.csv        (Milano, Torino: seeds 11/22/33)
  results/diagnostics/gini_cache_effect/gini_cache_effect_per_run_extra.csv  (Milano, Torino: seeds 44/55/66/77)
  results/diagnostics/gini_cache_effect/gini_cache_effect_per_run_genova60.csv (Genova: seeds 11/22/33, 60s budget)
  results/diagnostics/gini_cache_effect/multiseed_extra_per_run.csv         (Roma, Napoli, Genova: seeds 44/55/66/77)

Usage:
    python scripts/multi_seed_reliability_analysis.py
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
CR_DIR   = PROJECT_ROOT / "results" / "diagnostics" / "controlled_reconstruction"
GCE_DIR  = PROJECT_ROOT / "results" / "diagnostics" / "gini_cache_effect"
OUT_DIR  = GCE_DIR
D_COVER  = 2.0

CITY_CSV = {
    "Roma":   "center_432_Roma_k4400.csv",
    "Napoli": "center_547_Napoli_k3525.csv",
    "Genova": "center_276_Genova_k825.csv",
    "Torino": "center_207_Torino_k1950.csv",
    "Milano": "center_87_Milano_k3375.csv",
}
EXPECTED_SEEDS = {11, 22, 33, 44, 55, 66, 77}
N_BOOT = 5000
RNG_BOOT = np.random.default_rng(0)


def mean_arc_density(city_csv):
    """Structural property of site coords only — no seed/rng argument exists
    for this computation, so it is provably identical for every seed."""
    df = pd.read_csv(DATA_DIR / city_csv)
    coords_km = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float) * 111.0
    M = coords_km.shape[0]
    _, in_range, _, _ = build_arcs(coords_km, coords_km, D=D_COVER, forbid_self=False)
    degree = np.zeros(M)
    for (i, j) in in_range:
        degree[j] += 1
    return float(degree.mean())


def load_per_seed_deltas():
    frames = []

    cr = pd.read_csv(CR_DIR / "controlled_per_run.csv")
    cr = cr[cr["city"].isin(["Roma", "Napoli"]) & cr["label"].isin(["heap_only", "heap_cache"])]
    frames.append(cr[["city", "label", "seed", "best_score"]])

    gce = pd.read_csv(GCE_DIR / "gini_cache_effect_per_run.csv")
    gce = gce[gce["city"].isin(["Milano", "Torino"])]
    frames.append(gce[["city", "label", "seed", "best_score"]])

    extra = pd.read_csv(GCE_DIR / "gini_cache_effect_per_run_extra.csv")
    frames.append(extra[["city", "label", "seed", "best_score"]])

    genova = pd.read_csv(GCE_DIR / "gini_cache_effect_per_run_genova60.csv")
    frames.append(genova[["city", "label", "seed", "best_score"]])

    multiseed_path = GCE_DIR / "multiseed_extra_per_run.csv"
    if not multiseed_path.exists():
        raise FileNotFoundError(
            f"{multiseed_path} not found — run scripts/extend_multiseed_cache_effect.py first."
        )
    multiseed = pd.read_csv(multiseed_path)
    frames.append(multiseed[["city", "label", "seed", "best_score"]])

    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["city", "label", "seed"])
    piv = df.pivot_table(index=["city", "seed"], columns="label", values="best_score").reset_index()
    piv["delta"] = piv["heap_cache"] - piv["heap_only"]
    return piv[["city", "seed", "heap_only", "heap_cache", "delta"]]


def bootstrap_r(piv, arc_density, n_boot=N_BOOT, rng=RNG_BOOT):
    cities = list(arc_density.keys())
    ad = np.array([arc_density[c] for c in cities])
    seed_lists = {c: piv.loc[piv["city"] == c, "delta"].to_numpy() for c in cities}
    rs = np.empty(n_boot)
    for b in range(n_boot):
        means = np.array([
            rng.choice(seed_lists[c], size=len(seed_lists[c]), replace=True).mean()
            for c in cities
        ])
        rs[b] = np.corrcoef(ad, means)[0, 1]
    return rs


def main():
    piv = load_per_seed_deltas()

    # ── Verify every city has the same, expected seed set ──────────────────────
    print("Seed-coverage check (expect {11,22,33,44,55,66,77} for every city):")
    all_ok = True
    for city in CITY_CSV:
        seeds = set(piv.loc[piv["city"] == city, "seed"])
        ok = seeds == EXPECTED_SEEDS
        all_ok &= ok
        print(f"  {city:<8} seeds={sorted(seeds)}  {'OK' if ok else 'MISMATCH'}")
    if not all_ok:
        print("\nWARNING: not every city has the full 7-seed set — results below are still "
              "computed, but the pooled/bootstrap comparison is not apples-to-apples.\n")

    # ── Arc density: compute once per city, confirm it's seed-independent ──────
    arc_density = {city: mean_arc_density(csv) for city, csv in CITY_CSV.items()}
    print("\nMean arc density (structural, seed-independent by construction):")
    for city, ad in arc_density.items():
        print(f"  {city:<8} mean_degree={ad:7.2f}")

    piv["arc_density"] = piv["city"].map(arc_density)
    per_seed_csv = OUT_DIR / "multiseed_reliability_per_seed.csv"
    piv.sort_values(["city", "seed"]).to_csv(per_seed_csv, index=False)
    print(f"\nCSV -> {per_seed_csv}")

    # ── Per-city summary across seeds ───────────────────────────────────────────
    summary = (
        piv.groupby("city")["delta"]
           .agg(n_seeds="count", mean_delta="mean", std_delta="std",
                min_delta="min", max_delta="max")
           .reset_index()
    )
    summary["arc_density"] = summary["city"].map(arc_density)
    summary["se_delta"] = summary["std_delta"] / np.sqrt(summary["n_seeds"])
    summary["cv"] = (summary["std_delta"] / summary["mean_delta"].abs()).round(3)
    summary["sign_flip"] = summary.apply(
        lambda r: (r["min_delta"] < 0) and (r["max_delta"] > 0), axis=1
    )
    summary = summary.sort_values("arc_density").reset_index(drop=True)

    summary_csv = OUT_DIR / "multiseed_reliability_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"CSV -> {summary_csv}\n")
    print(summary[["city", "arc_density", "n_seeds", "mean_delta", "std_delta",
                    "cv", "sign_flip"]].to_string(index=False))

    flagged = summary[summary["sign_flip"] | (summary["cv"].abs() > 1.0)]
    if len(flagged):
        print(f"\nFLAGGED (sign flips across seeds, or |CV| > 1 -> single-seed result "
              f"for that city was likely noisy):")
        print(flagged[["city", "mean_delta", "std_delta", "cv", "sign_flip"]].to_string(index=False))
    else:
        print("\nNo city flagged for high seed-to-seed variance (no sign flips, |CV| <= 1 for all).")

    # ── Correlation: seed-averaged (n=5, like the original r=0.807 test) ───────
    r_seed_avg = summary["arc_density"].corr(summary["mean_delta"])
    print(f"\nPearson r, arc density vs SEED-AVERAGED delta (n={len(summary)} cities) = {r_seed_avg:.3f}")

    # ── Correlation: pooled, every seed its own point (n=35) ───────────────────
    r_pooled = piv["arc_density"].corr(piv["delta"])
    print(f"Pearson r, arc density vs delta, POOLED individual seeds (n={len(piv)} rows) = {r_pooled:.3f}")

    # ── Bootstrap over seed draws: how much does r move? ───────────────────────
    rs = bootstrap_r(piv, arc_density)
    r_lo, r_hi = np.percentile(rs, [2.5, 97.5])
    print(f"\nBootstrap (n={N_BOOT}, resample 7 seeds w/ replacement per city, recompute "
          f"seed-avg r each time):")
    print(f"  mean r = {rs.mean():.3f}   std = {rs.std():.3f}   95% CI = [{r_lo:.3f}, {r_hi:.3f}]")
    frac_below_0p5 = float((rs < 0.5).mean())
    frac_below_0 = float((rs < 0).mean())
    print(f"  P(bootstrap r < 0.5) = {frac_below_0p5:.3f}   P(bootstrap r < 0) = {frac_below_0:.3f}")

    verdict = (
        "CONCLUSION HOLDS: r stays well above 0.5 under seed resampling."
        if r_lo > 0.5 else
        "CONCLUSION WEAKENED: seed resampling pushes r below 0.5 in the lower tail "
        "of the bootstrap distribution — the original r=0.807 may be partly a "
        "lucky seed draw, not a stable structural relationship."
        if r_lo > 0 else
        "CONCLUSION UNDERMINED: seed resampling produces negative r in the lower "
        "tail — treat the arc-density predictor as unconfirmed until more cities "
        "or seeds are added."
    )
    print(f"\n{verdict}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.axhline(0, color="#888", linewidth=1.2, linestyle="--", zorder=1)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(summary)))
    color_map = dict(zip(summary["city"], colors))
    for city, c in color_map.items():
        sub = piv[piv["city"] == city]
        jitter = RNG_BOOT.uniform(-0.6, 0.6, len(sub))
        ax.scatter(sub["arc_density"] + jitter, sub["delta"], color=c, alpha=0.35,
                   s=45, zorder=2, edgecolors="none")
    for _, row in summary.iterrows():
        c = color_map[row["city"]]
        ax.errorbar(row["arc_density"], row["mean_delta"], yerr=row["se_delta"],
                    fmt="D", markersize=13, color=c, ecolor=c, capsize=5,
                    elinewidth=1.4, markeredgecolor="white", markeredgewidth=1.1, zorder=3)
        ax.annotate(f"{row['city']} (n={int(row['n_seeds'])})",
                    (row["arc_density"], row["mean_delta"]),
                    textcoords="offset points", xytext=(9, 6), fontsize=9)
    z = np.polyfit(summary["arc_density"], summary["mean_delta"], 1)
    xs = np.linspace(summary["arc_density"].min() - 5, summary["arc_density"].max() + 5, 50)
    ax.plot(xs, np.polyval(z, xs), color="#555", linewidth=1.3, linestyle=":",
            label=f"linear fit (r={r_seed_avg:.2f})")
    ax.set_xlabel("Mean arc density (# sites within 2.0 km)", fontsize=10.5)
    ax.set_ylabel("heap_cache $-$ heap_only best score\n(small dots = individual seeds, "
                  "diamonds = mean $\\pm$ SE)", fontsize=10)
    ax.set_title(f"Arc density vs. cache effect, n=7 seeds/city (r={r_seed_avg:.2f})", fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.2, linestyle="--")

    axR = axes[1]
    axR.hist(rs, bins=50, color="#2471A3", alpha=0.75, edgecolor="white", linewidth=0.4)
    axR.axvline(r_seed_avg, color="#C0392B", linewidth=1.8, linestyle="-",
                label=f"observed r={r_seed_avg:.2f}")
    axR.axvline(r_lo, color="#555", linewidth=1.2, linestyle="--", label=f"95% CI [{r_lo:.2f}, {r_hi:.2f}]")
    axR.axvline(r_hi, color="#555", linewidth=1.2, linestyle="--")
    axR.axvline(0.5, color="#888", linewidth=1.0, linestyle=":", label="r=0.5")
    axR.set_xlabel("Bootstrap Pearson r (arc density vs seed-avg delta)", fontsize=10.5)
    axR.set_ylabel("Count", fontsize=10.5)
    axR.set_title(f"Seed-resampling stability of r  ({N_BOOT} draws)", fontsize=11)
    axR.legend(fontsize=8.5, frameon=False)
    axR.grid(alpha=0.2, linestyle="--")

    fig.suptitle("Is the arc-density cache-effect predictor robust to seed choice?",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plot_path = OUT_DIR / "multiseed_reliability_check.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcdefaults()
    print(f"\nPlot -> {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
