"""
benchmark_largescale.py
=======================
Unified reconstruction-speedup benchmark for large-scale instances (N >= 1950).

Tests 4 configs in a single pass per city:

  1. baseline       — linear greedy, full recompute (no heap, no cache)
  2. old_R20        — linear greedy + 20% partial-delta cache (no heap)
  3. heap_full      — lazy-heap, full recompute (no cache)
  4. heap_R20_dw    — lazy-heap + 20% cache + distance-weighted refresh  ← full combo

All 4 use identical time budgets and seeds so speedups are directly comparable.

Usage:
    python scripts/benchmark_largescale.py              # all cities
    python scripts/benchmark_largescale.py --city Roma
    python scripts/benchmark_largescale.py --city Torino Milano
"""
import sys, io, time, argparse, csv as csv_mod
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evcs.geom         import build_arcs
from evcs.destroy      import destroy_multi_u
from evcs.greedy       import reconstruct_u_dict_fast
from evcs.methods      import greedy_schedule_multi_from_variants
from evcs.ue_evaluator import UEEvaluator

DATA_DIR = PROJECT_ROOT / "data" / "input"
OUT_DIR  = PROJECT_ROOT / "results" / "largescale"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters (match existing benchmarks) ─────────────────────────────
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

# ── Large-scale instances only ────────────────────────────────────────────────
INSTANCES = {
    "Torino": ("center_207_Torino_k1950.csv", 1950, 360),
    "Milano": ("center_87_Milano_k3375.csv",  3375, 600),
    "Napoli": ("center_547_Napoli_k3525.csv", 3525, 660),
    "Roma":   ("center_432_Roma_k4400.csv",   4400, 780),
}

# ── 4 configs: (label, refresh_R, use_heap, use_dist_weight) ─────────────────
CONFIGS = [
    ("baseline",    1.0,  False, False),   # full recompute, linear scan
    ("old_R20",     0.20, False, False),   # cache only, no heap, no dist-weight
    ("heap_full",   1.0,  True,  False),   # heap only, full recompute
    ("heap_R20_dw", 0.20, True,  True),    # full combo: heap + cache + dist-weight
]


def load_instance(csv_name):
    df        = pd.read_csv(DATA_DIR / csv_name)
    coords_km = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float) * 111.0
    pop       = np.maximum(df["Aggregated_Population"].to_numpy(float), 0.0)
    M         = coords_km.shape[0]

    base_d    = (pop / pop.sum()) * M
    rng_inst  = np.random.default_rng(SEED)
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


def run_config(label, refresh_R, use_dist_weight,
               inst, Ij_int, Ji_int, P_T, ue_fn, budget_s):
    rng         = np.random.default_rng(SEED)
    M           = inst["M"]
    use_cache   = refresh_R < 1.0
    delta_cache = {} if use_cache else None
    site_coords = inst["coords_J"] if use_dist_weight else None

    U_curr      = greedy_schedule_multi_from_variants(inst, P_T, D_COVER, seed=SEED)
    score_curr  = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

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
        )
        t_reconstruct += time.perf_counter() - _t

        _t = time.perf_counter()
        score_recon = ue_fn(U_recon)
        t_ue += time.perf_counter() - _t
        n_evals += 1

        if score_recon > score_best:
            score_best, U_best = score_recon, dict(U_recon)
        U_curr = dict(U_recon)
        n_iters += 1

    t_elapsed = time.perf_counter() - t0
    return {
        "label":           label,
        "refresh_R":       refresh_R,
        "dist_weighted":   use_dist_weight,
        "iters":           n_iters,
        "evals":           n_evals,
        "best_score":      round(score_best, 3),
        "t_reconstruct_s": round(t_reconstruct, 2),
        "t_ue_s":          round(t_ue, 2),
        "t_destroy_s":     round(t_destroy, 2),
        "pct_reconstruct": round(100 * t_reconstruct / t_elapsed, 1),
        "pct_ue":          round(100 * t_ue / t_elapsed, 1),
        "avg_recon_ms":    round(1000 * t_reconstruct / max(n_iters, 1), 1),
        "avg_ue_ms":       round(1000 * t_ue / max(n_evals, 1), 1),
    }


def run_city(city):
    csv_name, N, budget_s = INSTANCES[city]
    print(f"\n{'='*70}")
    print(f"  {city}   N={N}   budget={budget_s}s/config   seed={SEED}")
    print(f"{'='*70}")
    print("  Loading instance...", end=" ", flush=True)
    inst, Ij_int, Ji_int, P_T, ue_fn = load_instance(csv_name)
    print("done")

    results = []
    baseline_ms = None

    for label, R, use_heap, use_dw in CONFIGS:
        print(f"\n  [{label}]  R={R}  heap={use_heap}  dist_weighted={use_dw}")
        r = run_config(label, R, use_dw, inst, Ij_int, Ji_int, P_T, ue_fn, budget_s)
        r["city"] = city
        r["N"]    = N

        if baseline_ms is None:
            baseline_ms = r["avg_recon_ms"]

        speedup = round(baseline_ms / max(r["avg_recon_ms"], 0.01), 2)
        r["speedup_vs_baseline"] = speedup
        results.append(r)

        print(f"    iters={r['iters']}  best={r['best_score']:.3f}  "
              f"recon={r['avg_recon_ms']:.1f} ms/iter ({r['pct_reconstruct']:.1f}%)  "
              f"speedup={speedup:.2f}x")

    # ── Per-city summary ──────────────────────────────────────────────────────
    print(f"\n  {'─'*70}")
    print(f"  {'Config':<14} {'Iters':>6} {'Best':>10} {'Recon ms':>10} "
          f"{'Speedup':>10} {'Recon%':>8}")
    print(f"  {'─'*70}")
    for r in results:
        print(f"  {r['label']:<14} {r['iters']:>6} {r['best_score']:>10.3f} "
              f"{r['avg_recon_ms']:>10.1f} {r['speedup_vs_baseline']:>9.2f}x "
              f"{r['pct_reconstruct']:>7.1f}%")
    print(f"  {'─'*70}")

    out_csv = OUT_DIR / f"{city}_largescale_results.csv"
    fields  = ["city", "N", "label", "refresh_R", "dist_weighted", "iters", "evals",
               "best_score", "t_reconstruct_s", "t_ue_s", "t_destroy_s",
               "pct_reconstruct", "pct_ue", "avg_recon_ms", "avg_ue_ms",
               "speedup_vs_baseline"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv_mod.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\n  CSV -> {out_csv}")
    return results


# ── Entry point ───────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--city", nargs="+", default=None,
                choices=list(INSTANCES.keys()),
                help="One or more cities (default: all)")
args = ap.parse_args()

cities = args.city if args.city else list(INSTANCES.keys())

all_results = []
for c in cities:
    all_results.extend(run_city(c))

# ── Cross-city summary table ──────────────────────────────────────────────────
print(f"\n\n{'='*78}")
print("  LARGE-SCALE SPEEDUP SUMMARY")
print(f"{'='*78}")
print(f"  {'City':<10} {'N':>5}  "
      f"{'baseline':>10}  {'old_R20':>10}  {'heap_full':>10}  {'heap_R20_dw':>12}")
print(f"  {'':10} {'':>5}  "
      f"{'ms/iter':>10}  {'ms / ×':>10}  {'ms / ×':>10}  {'ms / ×':>12}")
print(f"  {'─'*78}")

by_city = {}
for r in all_results:
    by_city.setdefault(r["city"], {})[r["label"]] = r

for city in cities:
    d  = by_city[city]
    bms = d["baseline"]["avg_recon_ms"]
    def fmt(lbl):
        ms = d[lbl]["avg_recon_ms"]
        sp = d[lbl]["speedup_vs_baseline"]
        return f"{ms:7.1f} /{sp:5.1f}x"
    print(f"  {city:<10} {d['baseline']['N']:>5}  "
          f"{bms:>10.1f}  {fmt('old_R20')}  {fmt('heap_full')}  {fmt('heap_R20_dw')}")

print(f"  {'─'*78}")

# Save combined CSV
combined_csv = OUT_DIR / "largescale_summary.csv"
all_df = pd.DataFrame(all_results)
all_df.to_csv(combined_csv, index=False)
print(f"\n  Combined CSV -> {combined_csv}")
print("\nDone.")
