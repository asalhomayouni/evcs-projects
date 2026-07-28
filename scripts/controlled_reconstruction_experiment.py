"""
controlled_reconstruction_experiment.py
========================================
Controlled, multi-seed comparison of cache-only / heap-only / heap+cache
reconstruction on Napoli vs Roma, to determine whether the "Napoli anomaly"
seen in the single-seed benchmark_largescale.py run (heap/cache configs not
clearly beating baseline the way they do for other cities) is an algorithmic
effect (would appear identically on any instance with this geometry / demand
shape) or is driven by seed-to-seed noise in one particular random draw.

Key control: P_t = floor(N_ref / 30) is IDENTICAL for both cities (N_ref is
fixed, not each city's own N), so any difference between Napoli and Roma
can't be explained by them simply being given a different per-period budget.

Both demand realisation and search randomness are re-drawn from each seed
(not just the search), so a pattern that survives across seeds reflects the
city's underlying demand/geometry, not one lucky/unlucky draw.

Configs (see use_heap in evcs.greedy.reconstruct_u_dict_fast):
  cache_only  — refresh_ratio=0.20, dist-weighted, heap OFF (linear rescan)
  heap_only   — refresh_ratio=1.00 (no cache),      heap ON
  heap_cache  — refresh_ratio=0.20, dist-weighted,  heap ON   (full combo)

Usage:
    python scripts/controlled_reconstruction_experiment.py
    python scripts/controlled_reconstruction_experiment.py --seeds 11 22 33 --budget 120
    python scripts/controlled_reconstruction_experiment.py --city Napoli Roma --nref 3525
"""
import sys, io, time, argparse, csv as csv_mod
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evcs.geom          import build_arcs
from evcs.destroy       import destroy_multi_u
from evcs.greedy        import reconstruct_u_dict_fast
from evcs.methods       import greedy_schedule_multi_from_variants
from evcs.ue_evaluator  import UEEvaluator

DATA_DIR = PROJECT_ROOT / "data" / "input"
OUT_DIR  = PROJECT_ROOT / "results" / "diagnostics" / "controlled_reconstruction"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fixed hyper-parameters (match benchmark_largescale.py) ───────────────────
T_PERIODS    = 6
D_COVER      = 2.0
MAX_CHARGERS = 6
Q_CAP        = 20.0
CUMULATIVE   = True
FRAC_REMOVE  = 0.20
UE_ALPHA     = 10.0
UE_NOOPT     = 10.0
UE_PEN_DIST  = 5.0
UE_MU_EXTRA  = 1e-4
UE_CAP_SRV   = 4.0
UE_MAX_RANGE = 2.0
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]

INSTANCES = {
    "Torino": "center_207_Torino_k1950.csv",
    "Milano": "center_87_Milano_k3375.csv",
    "Napoli": "center_547_Napoli_k3525.csv",
    "Roma":   "center_432_Roma_k4400.csv",
}

# (label, refresh_R, use_heap, use_dist_weight)
CONFIGS = [
    ("cache_only", 0.20, False, True),
    ("heap_only",  1.00, True,  False),
    ("heap_cache", 0.20, True,  True),
]


def load_instance(csv_name, seed, P_T):
    df        = pd.read_csv(DATA_DIR / csv_name)
    coords_km = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float) * 111.0
    pop       = np.maximum(df["Aggregated_Population"].to_numpy(float), 0.0)
    M         = coords_km.shape[0]

    base_d    = (pop / pop.sum()) * M
    rng_inst  = np.random.default_rng(seed)
    demand_TM = np.zeros((T_PERIODS, M))
    for t in range(T_PERIODS):
        sf = 1.0 + 0.2 * np.sin(2 * np.pi * t / T_PERIODS)
        demand_TM[t] = np.maximum(0.0, base_d * sf * (1.0 + rng_inst.normal(0, 0.05, M)))

    distIJ, in_range, Ji_ba, Ij_ba = build_arcs(
        coords_km, coords_km, D=D_COVER, forbid_self=False
    )
    Ij_int = {j: [] for j in range(M)}
    Ji_int = {i: [] for i in range(M)}
    for (i, j) in in_range:
        Ij_int[j].append(int(i))
        Ji_int[i].append(int(j))

    inst = {
        "coords_I":  coords_km,
        "coords_J":  coords_km.copy(),
        "demand_IT": demand_TM,
        "distIJ":    distIJ,
        "in_range":  in_range,
        "Ji":        Ji_ba,
        "Ij":        Ij_ba,
        "M":         M,
        "N":         M,
    }

    ue_eval = UEEvaluator(
        N=M, d=demand_TM[0], tau=distIJ,
        mu=UE_CAP_SRV + UE_MU_EXTRA, s_max=MAX_CHARGERS,
        noopt_cost=UE_NOOPT, alpha_wait=UE_ALPHA, N_bp=50,
        max_range=UE_MAX_RANGE, penalty_distance=UE_PEN_DIST,
    )

    def ue_fn(U):
        s, _ = ue_eval.evaluate(U, T=T_PERIODS, cumulative_install=CUMULATIVE)
        return float(s)

    return inst, Ij_int, ue_eval, ue_fn, M


def run_config(label, refresh_R, use_heap, use_dist_weight,
               inst, Ij_int, P_T, ue_fn, budget_s, seed):
    rng         = np.random.default_rng(seed)
    use_cache   = refresh_R < 1.0
    delta_cache = {} if use_cache else None
    site_coords = inst["coords_J"] if use_dist_weight else None

    U_curr      = greedy_schedule_multi_from_variants(inst, P_T, D_COVER, seed=seed)
    score_curr  = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    t_reconstruct = 0.0
    n_iters = 0
    t0 = time.perf_counter()

    while time.perf_counter() - t0 < budget_s:
        base = U_curr if rng.random() < 0.7 else U_best
        mode = str(rng.choice(DESTROY_MODES))

        U_partial, _ = destroy_multi_u(
            base, inst, rng, P_T, FRAC_REMOVE, mode,
            site_cap=MAX_CHARGERS, cumulative_install=CUMULATIVE,
        )

        destroyed_js = None
        if use_cache:
            destroyed_js = {int(j) for (j, _tp), v_b in base.items()
                            if int(v_b) > int(U_partial.get((j, _tp), 0))}

        _t = time.perf_counter()
        U_recon, _ = reconstruct_u_dict_fast(
            U_partial, inst["demand_IT"], P_T, Ij_int,
            U_cap=MAX_CHARGERS, Q=Q_CAP, rng=rng,
            cumulative_install=CUMULATIVE,
            delta_cache=delta_cache,
            refresh_ratio=refresh_R,
            destroyed_js=destroyed_js,
            site_coords=site_coords,
            use_heap=use_heap,
        )
        t_reconstruct += time.perf_counter() - _t

        score_recon = ue_fn(U_recon)

        if score_recon > score_best:
            score_best, U_best = score_recon, dict(U_recon)
        U_curr = dict(U_recon)
        n_iters += 1

    t_elapsed = time.perf_counter() - t0
    return {
        "label":           label,
        "refresh_R":       refresh_R,
        "use_heap":        use_heap,
        "dist_weighted":   use_dist_weight,
        "seed":            seed,
        "iters":           n_iters,
        "best_score":      round(score_best, 3),
        "t_reconstruct_s": round(t_reconstruct, 2),
        "avg_recon_ms":    round(1000 * t_reconstruct / max(n_iters, 1), 2),
    }


def run_city(city, csv_name, P_T, seeds, budget_s):
    print(f"\n{'='*70}")
    print(f"  {city}   P_t={P_T[0]} (fixed N_ref budget)   "
          f"budget={budget_s}s/config/seed   seeds={seeds}")
    print(f"{'='*70}")

    results = []
    for seed in seeds:
        print(f"  Loading instance (seed={seed})...", end=" ", flush=True)
        inst, Ij_int, ue_eval, ue_fn, M = load_instance(csv_name, seed, P_T)
        print(f"done  N={M}")

        for label, R, use_heap, use_dw in CONFIGS:
            r = run_config(label, R, use_heap, use_dw, inst, Ij_int, P_T,
                            ue_fn, budget_s, seed)
            r["city"] = city
            r["N"]    = M
            results.append(r)
            print(f"    [seed={seed}] {label:<11} iters={r['iters']:>4}  "
                  f"best={r['best_score']:>9.3f}  recon={r['avg_recon_ms']:>7.2f} ms/iter")

        ue_eval.dispose()

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", default=["Napoli", "Roma"],
                     choices=list(INSTANCES.keys()))
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--budget", type=float, default=120.0,
                     help="seconds per (city, config, seed)")
    ap.add_argument("--nref", type=int, default=3525,
                     help="fixed N_ref for P_t = floor(N_ref/30), same for all cities")
    args = ap.parse_args()

    P_t = max(4, args.nref // 30)
    P_T = [P_t] * T_PERIODS
    print(f"N_ref={args.nref}  ->  P_t={P_t} chargers/period for ALL cities "
          f"(controlled budget, not city-specific)")

    all_results = []
    for city in args.city:
        all_results.extend(
            run_city(city, INSTANCES[city], P_T, args.seeds, args.budget)
        )

    df = pd.DataFrame(all_results)
    per_run_csv = OUT_DIR / "controlled_per_run.csv"
    df.to_csv(per_run_csv, index=False)
    print(f"\nCSV -> {per_run_csv}")

    # ── Summary: mean/std across seeds per (city, config) ────────────────────
    summary = (
        df.groupby(["city", "label"])
          .agg(mean_score=("best_score", "mean"),
               std_score=("best_score", "std"),
               min_score=("best_score", "min"),
               max_score=("best_score", "max"),
               mean_iters=("iters", "mean"),
               mean_recon_ms=("avg_recon_ms", "mean"))
          .round(3)
          .reset_index()
    )
    summary_csv = OUT_DIR / "controlled_summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"CSV -> {summary_csv}\n")
    print(summary.to_string(index=False))

    # ── Noise vs. algorithmic-effect diagnostic ───────────────────────────────
    # within-config seed spread (noise) vs across-config spread at fixed seed
    # (algorithmic effect). If Napoli's within-config std rivals or exceeds its
    # across-config spread, the "anomaly" is more likely noise; if the config
    # ranking is stable across seeds and the across-config spread dominates,
    # it's a repeatable property of the instance.
    print(f"\n{'─'*70}")
    print("  NOISE vs ALGORITHMIC-EFFECT DIAGNOSTIC")
    print(f"{'─'*70}")
    for city in args.city:
        sub = df[df["city"] == city]
        within_std = sub.groupby("label")["best_score"].std().mean()
        by_seed_range = sub.groupby("seed")["best_score"].apply(lambda s: s.max() - s.min())
        across_config_spread = by_seed_range.mean()
        ratio = within_std / max(across_config_spread, 1e-9)
        verdict = "NOISE-DOMINATED (likely seed artifact)" if ratio > 0.7 else \
                  "STABLE ACROSS SEEDS (likely instance-specific)"
        print(f"  {city:<8}  mean within-config seed-std={within_std:6.3f}  "
              f"mean across-config spread={across_config_spread:6.3f}  "
              f"ratio={ratio:5.2f}  -> {verdict}")

    # ── Plot: box + strip per config, side by side by city ───────────────────
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    labels = [c[0] for c in CONFIGS]
    colors = {"cache_only": "#2471A3", "heap_only": "#C0392B", "heap_cache": "#27AE60"}

    fig, axes = plt.subplots(1, len(args.city), figsize=(6 * len(args.city), 5.5),
                              sharey=False)
    if len(args.city) == 1:
        axes = [axes]
    fig.suptitle(
        f"Controlled reconstruction experiment  (P_t={P_t}, N_ref={args.nref}, "
        f"{len(args.seeds)} seeds)",
        fontsize=12, y=1.02,
    )
    for ax, city in zip(axes, args.city):
        sub = df[df["city"] == city]
        for xi, label in enumerate(labels):
            scores = sub[sub["label"] == label]["best_score"].to_numpy()
            c = colors[label]
            ax.boxplot(scores, positions=[xi], widths=0.45, patch_artist=True,
                       manage_ticks=False,
                       boxprops=dict(facecolor=c, alpha=0.45, linewidth=1.2),
                       medianprops=dict(color="black", linewidth=2.0))
            jitter = np.random.default_rng(42).uniform(-0.10, 0.10, len(scores))
            ax.scatter([xi + j for j in jitter], scores, color=c, s=55,
                       zorder=4, edgecolors="white", linewidths=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Best UE score", fontsize=10)
        n_city = int(sub["N"].iloc[0])
        ax.set_title(f"{city}  (N={n_city})", fontsize=11)
        ax.grid(axis="y", alpha=0.2, linestyle="--")

    plt.tight_layout()
    plot_path = OUT_DIR / "controlled_reconstruction_boxplot.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    plt.rcdefaults()
    print(f"\nPlot -> {plot_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
