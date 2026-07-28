"""
benchmark_lazy_heap.py
Measures the combined speedup of:
  (1) lazy greedy heap  — avoids rescanning all N sites after each placement
  (2) between-iteration partial delta cache  — R% refresh per D&R call
  (3) distance-weighted R% sampling  — prioritise sites near destroyed region

Configurations tested:
  heap+full   — heap + full refresh (R=1.0, like the old no-cache approach
                but now with heap benefit at steps 1+)
  heap+R=20%  — heap + 20% dist-weighted cache
  heap+R=10%  — heap + 10% dist-weighted cache  (expected sweet-spot)
  heap+R=5%   — heap + 5% dist-weighted cache

Usage:
    python scripts/benchmark_lazy_heap.py --city Genova
    python scripts/benchmark_lazy_heap.py --city Torino
    python scripts/benchmark_lazy_heap.py --city Milano
    python scripts/benchmark_lazy_heap.py --all
"""
import sys, io, time, argparse, csv as csv_mod
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evcs.geom          import build_arcs
from evcs.destroy       import destroy_multi_u
from evcs.greedy        import reconstruct_u_dict_fast
from evcs.methods       import greedy_schedule_multi_from_variants
from evcs.ue_evaluator  import UEEvaluator

DATA_DIR = PROJECT_ROOT / "data" / "input"
OUT_DIR  = PROJECT_ROOT / "results" / "gate" / "lazy_heap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Fixed hyper-parameters ────────────────────────────────────────────────────
T_PERIODS    = 6
D_COVER      = 2.0
MAX_CHARGERS = 6
Q_CAP        = 20.0
CUMULATIVE   = True
FRAC_REMOVE  = 0.20
SEED         = 11
UE_ALPHA     = 10.0
UE_NOOPT     = 10.0
UE_PEN_DIST  = 5.0
UE_MU_EXTRA  = 1e-4
UE_CAP_SRV   = 4.0
UE_MAX_RANGE = 2.0

DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]

INSTANCES = {
    "Verona": ("center_146_Verona_k250.csv",   250,   90),
    "Genova": ("center_276_Genova_k825.csv",   825,  180),
    "Torino": ("center_207_Torino_k1950.csv", 1950,  360),
    "Milano": ("center_87_Milano_k3375.csv",  3375,  600),
    "Napoli": ("center_547_Napoli_k3525.csv", 3525,  660),
    "Roma":   ("center_432_Roma_k4400.csv",   4400,  780),
}

# (label, refresh_ratio, use_dist_weight)
CONFIGS = [
    ("heap+full",  1.0,  False),   # heap + full refresh — new baseline
    ("heap+R=20%", 0.20, True),
    ("heap+R=10%", 0.10, True),
    ("heap+R=5%",  0.05, True),
]


def load_instance(csv_name):
    df = pd.read_csv(DATA_DIR / csv_name)
    coords_km = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float) * 111.0
    pop       = np.maximum(df["Aggregated_Population"].to_numpy(float), 0.0)
    M         = coords_km.shape[0]

    base_d   = (pop / pop.sum()) * M
    rng_inst = np.random.default_rng(SEED)
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

    P_T = [max(4, M // 30)] * T_PERIODS

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

    return inst, Ij_int, Ji_int, P_T, ue_fn


def run_config(label, refresh_ratio, use_dist_weight,
               inst, Ij_int, Ji_int, P_T, ue_fn, budget_s):
    rng = np.random.default_rng(SEED)
    M   = inst["M"]
    site_coords = inst["coords_J"] if use_dist_weight else None

    U_curr     = greedy_schedule_multi_from_variants(inst, P_T, D_COVER, seed=SEED)
    score_curr = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    use_cache   = refresh_ratio < 1.0
    delta_cache = {} if use_cache else None

    t_destroy = t_reconstruct = t_ue = 0.0
    n_iters = n_evals = 0

    t0 = time.perf_counter()

    while time.perf_counter() - t0 < budget_s:
        base = U_curr if rng.random() < 0.7 else U_best
        mode = str(rng.choice(DESTROY_MODES))

        _t = time.perf_counter()
        U_partial, _ = destroy_multi_u(
            base, inst, rng, P_T, FRAC_REMOVE, mode,
            site_cap=MAX_CHARGERS, cumulative_install=CUMULATIVE,
        )
        t_destroy += time.perf_counter() - _t

        if use_cache:
            destroyed_js = {int(j) for (j, _tp), v_b in base.items()
                            if int(v_b) > int(U_partial.get((j, _tp), 0))}
        else:
            destroyed_js = None

        _t = time.perf_counter()
        U_recon, _ = reconstruct_u_dict_fast(
            U_partial, inst["demand_IT"], P_T, Ij_int,
            U_cap=MAX_CHARGERS, Q=Q_CAP, rng=rng,
            cumulative_install=CUMULATIVE,
            delta_cache=delta_cache,
            refresh_ratio=refresh_ratio,
            destroyed_js=destroyed_js,
            site_coords=site_coords,
        )
        t_reconstruct += time.perf_counter() - _t

        _t = time.perf_counter()
        score_recon = ue_fn(U_recon)
        t_ue += time.perf_counter() - _t
        n_evals += 1

        if score_recon > score_best:
            score_best = score_recon
            U_best     = dict(U_recon)

        U_curr = dict(U_recon)
        n_iters += 1

    t_elapsed = time.perf_counter() - t0

    return {
        "label":           label,
        "refresh_R":       refresh_ratio,
        "dist_weighted":   use_dist_weight,
        "iters":           n_iters,
        "evals":           n_evals,
        "best_score":      round(score_best, 3),
        "t_reconstruct_s": round(t_reconstruct, 2),
        "t_ue_s":          round(t_ue, 2),
        "t_destroy_s":     round(t_destroy, 2),
        "pct_reconstruct": round(100 * t_reconstruct / t_elapsed, 1),
        "pct_ue":          round(100 * t_ue          / t_elapsed, 1),
        "avg_recon_ms":    round(1000 * t_reconstruct / max(n_iters, 1), 1),
        "avg_ue_ms":       round(1000 * t_ue          / max(n_evals, 1), 1),
    }


def run_city(city):
    csv_name, N, budget_s = INSTANCES[city]
    print(f"\n{'='*70}")
    print(f"  {city}   N={N}   budget={budget_s}s   seed={SEED}")
    print(f"  lazy-greedy heap + dist-weighted partial-delta cache")
    print(f"{'='*70}")

    print("  Loading instance...", end=" ", flush=True)
    inst, Ij_int, Ji_int, P_T, ue_fn = load_instance(csv_name)
    print("done")

    results = []
    base_score = base_ms = None

    for label, R, dw in CONFIGS:
        print(f"\n  [{label}]  R={R}  dist_weighted={dw}")
        r = run_config(label, R, dw, inst, Ij_int, Ji_int, P_T, ue_fn, budget_s)
        r["city"] = city
        r["N"]    = N
        results.append(r)

        if base_score is None:
            base_score = r["best_score"]
            base_ms    = r["avg_recon_ms"]

        d_score = round(r["best_score"] - base_score, 3)
        speedup = round(base_ms / max(r["avg_recon_ms"], 0.01), 2)

        print(f"    iters={r['iters']}  best={r['best_score']:.3f}  "
              f"Δ_vs_heap_full={d_score:+.3f}")
        print(f"    reconstruct {r['avg_recon_ms']:.1f} ms/iter "
              f"({r['pct_reconstruct']:.1f}%)  speedup={speedup:.2f}x")
        print(f"    UE          {r['avg_ue_ms']:.1f} ms/eval "
              f"({r['pct_ue']:.1f}%)")

    print(f"\n  {'─'*70}")
    print(f"  {'Config':<14} {'Iters':>6} {'Best':>10} "
          f"{'Δscore':>9} {'Recon ms':>10} {'Speedup':>9} {'Recon%':>8}")
    print(f"  {'─'*70}")
    for r in results:
        d  = round(r["best_score"] - base_score, 3)
        sp = round(base_ms / max(r["avg_recon_ms"], 0.01), 2)
        print(f"  {r['label']:<14} {r['iters']:>6} {r['best_score']:>10.3f} "
              f"{d:>+9.3f} {r['avg_recon_ms']:>10.1f} {sp:>9.2f}x "
              f"{r['pct_reconstruct']:>7.1f}%")
    print(f"  {'─'*70}")

    out_csv = OUT_DIR / f"{city}_lazy_heap_results.csv"
    fnames  = ["city", "N", "label", "refresh_R", "dist_weighted", "iters", "evals",
               "best_score", "t_reconstruct_s", "t_ue_s", "t_destroy_s",
               "pct_reconstruct", "pct_ue", "avg_recon_ms", "avg_ue_ms"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.DictWriter(f, fieldnames=fnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\n  CSV -> {out_csv}")
    return results


ap = argparse.ArgumentParser()
ap.add_argument("--city", default=None, choices=list(INSTANCES.keys()))
ap.add_argument("--all",  action="store_true")
args = ap.parse_args()

cities = list(INSTANCES.keys()) if args.all else ([args.city] if args.city else ["Genova"])
for c in cities:
    run_city(c)
print("\nDone.")
