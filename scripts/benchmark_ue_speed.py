"""
UE evaluator speed benchmark across instance sizes.

Instances tested (by ascending N):
  Vicenza  N=125   center_102_Vicenza_k125.csv
  Verona   N=250   center_146_Verona_k250.csv
  Monza    N=400   center_79_Monza_k400.csv

For each instance: generate N_SOLUTIONS random solutions via destroy-reconstruct,
time each UE evaluation individually, then report timing statistics.

Usage:
    python scripts/benchmark_ue_speed.py
    python scripts/benchmark_ue_speed.py --n-solutions 100
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evcs.geom import build_arcs
from evcs.destroy import destroy_multi_u
from evcs.greedy import reconstruct_u_dict_fast
from evcs.methods import greedy_schedule_multi_from_variants
from evcs.ue_evaluator import UEEvaluator

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--n-solutions",         type=int,   default=50,
               help="UE evaluations per instance (default 50)")
p.add_argument("--seed",                type=int,   default=11)
p.add_argument("--D",                   type=float, default=2.0)
p.add_argument("--T",                   type=int,   default=6)
p.add_argument("--ue-alpha-wait",       type=float, default=10.0)
p.add_argument("--ue-noopt",            type=float, default=30.0)
p.add_argument("--ue-max-range",        type=float, default=2.0)
p.add_argument("--ue-penalty-distance", type=float, default=5.0)
p.add_argument("--ue-cap-per-server",   type=float, default=4.0)
args = p.parse_args()

DATA_DIR    = PROJECT_ROOT / "data" / "input"
RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INSTANCES = [
    ("Vicenza",  "center_102_Vicenza_k125.csv"),
    ("Verona",   "center_146_Verona_k250.csv"),
    ("Monza",    "center_79_Monza_k400.csv"),
]

# ── Helper: build instance ────────────────────────────────────────────────────
def build_instance(csv_path, T, seed, D):
    df = pd.read_csv(csv_path)
    coords_deg = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float)
    coords_km  = coords_deg.copy()
    coords_km[:, 0] *= 111.0
    coords_km[:, 1] *= 111.0

    pop       = np.maximum(df["Aggregated_Population"].to_numpy(float), 0.0)
    pop_share = pop / pop.sum()
    M         = coords_km.shape[0]
    base      = pop_share * M

    rng_inst  = np.random.default_rng(seed)
    demand_TM = np.zeros((T, M))
    for t in range(T):
        sf = 1.0 + 0.2 * np.sin(2 * np.pi * t / T)
        demand_TM[t] = np.maximum(0.0, base * sf * (1.0 + rng_inst.normal(0, 0.05, M)))

    distIJ, in_range, Ji, Ij = build_arcs(coords_km, coords_km, D=D, forbid_self=False)

    Ij_int = {j: [] for j in range(M)}
    Ji_int = {i: [] for i in range(M)}
    for (i, j) in in_range:
        Ij_int[j].append(i)
        Ji_int[i].append(j)

    inst = {
        "coords_I": coords_km, "coords_J": coords_km.copy(),
        "demand_IT": demand_TM, "distIJ": distIJ,
        "in_range": in_range, "Ji": Ji, "Ij": Ij, "M": M, "N": M,
    }
    return inst, demand_TM, distIJ, Ij_int, M


# ── Benchmark loop ────────────────────────────────────────────────────────────
results = []   # list of dicts per instance

sep = "=" * 65
print(f"\n{sep}")
print(f"UE speed benchmark  ({args.n_solutions} solutions per instance)")
print(f"UE params: max_range={args.ue_max_range}km  alpha_wait={args.ue_alpha_wait}  "
      f"pd={args.ue_penalty_distance}  cap/s={args.ue_cap_per_server}")
print(sep)

for city, csv_file in INSTANCES:
    csv_path = DATA_DIR / csv_file
    print(f"\n[{city}]  {csv_file}")

    inst, demand_TM, distIJ, Ij_int, M = build_instance(
        csv_path, args.T, args.seed, args.D
    )
    T = args.T
    P_T               = [8] * T
    max_chargers      = 6
    Q_cap             = 20.0
    cumulative_install = True
    destroy_modes     = ["site_swap", "local_remove", "area_destroy"]

    print(f"  N={M}  T={T}  D={args.D}  |arcs|={len(inst['in_range'])}")

    ue_cap = args.ue_cap_per_server
    ue_mu  = ue_cap + 1e-4

    ue_eval = UEEvaluator(
        N=M, d=demand_TM[0], tau=distIJ, mu=ue_mu, s_max=max_chargers,
        noopt_cost=args.ue_noopt, alpha_wait=args.ue_alpha_wait, N_bp=50,
        max_range=args.ue_max_range, penalty_distance=args.ue_penalty_distance,
    )

    rng    = np.random.default_rng(args.seed + 99)
    U_base = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=args.seed)

    # Generate solutions first (don't count D-R time in UE timing)
    solutions = []
    for _ in range(args.n_solutions):
        frac = rng.uniform(0.3, 0.7)
        mode = str(rng.choice(destroy_modes))
        U_partial, _ = destroy_multi_u(
            U_base, inst, rng, P_T, frac, mode,
            site_cap=max_chargers, cumulative_install=cumulative_install,
        )
        U_sol, _ = reconstruct_u_dict_fast(
            U_partial, demand_TM, P_T, Ij_int,
            U_cap=max_chargers, Q=Q_cap, rng=rng,
            cumulative_install=cumulative_install,
        )
        solutions.append(U_sol)

    # Time each UE eval individually (warmup: skip first 3 for JIT/cache effects)
    times = []
    scores = []
    for idx, U_sol in enumerate(solutions):
        t_start = time.perf_counter()
        score, _ = ue_eval.evaluate(U_sol, T=T, cumulative_install=cumulative_install)
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        if idx >= 3:    # skip first 3 as warmup
            times.append(elapsed)
            scores.append(float(score))

    times = np.array(times)
    scores = np.array(scores)

    n_effective = len(times)
    mean_t  = times.mean()
    std_t   = times.std()
    p50_t   = float(np.percentile(times, 50))
    p95_t   = float(np.percentile(times, 95))
    p99_t   = float(np.percentile(times, 99))
    total_t = times.sum()

    # At LS scale: how many steps per second? How long for a full LS run (8 moves)?
    ls_moves    = 8
    ls_run_time = mean_t * ls_moves

    print(f"  Timing ({n_effective} evals, first 3 excluded as warmup):")
    print(f"    mean={mean_t*1000:.1f}ms   std={std_t*1000:.1f}ms   "
          f"p50={p50_t*1000:.1f}ms   p95={p95_t*1000:.1f}ms   p99={p99_t*1000:.1f}ms")
    print(f"    Total for {n_effective} evals: {total_t:.2f}s")
    print(f"    At LS scale: ~{ls_run_time:.2f}s per {ls_moves}-move LS run  "
          f"({1/mean_t:.1f} evals/s)")
    print(f"  UE scores: mean={scores.mean():.3f}  std={scores.std():.3f}  "
          f"range={scores.max()-scores.min():.3f}")

    results.append({
        "city":        city,
        "csv":         csv_file,
        "N":           M,
        "n_arcs":      len(inst["in_range"]),
        "n_evals":     n_effective,
        "mean_ms":     mean_t * 1000,
        "std_ms":      std_t  * 1000,
        "p50_ms":      p50_t  * 1000,
        "p95_ms":      p95_t  * 1000,
        "p99_ms":      p99_t  * 1000,
        "evals_per_s": 1.0 / mean_t,
        "ls8_run_s":   ls_run_time,
        "ue_score_std": float(scores.std()),
    })

    ue_eval.dispose()

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("Summary table:")
print(f"  {'City':<10} {'N':>6} {'|arcs|':>8} {'mean(ms)':>10} {'p95(ms)':>10} "
      f"{'evals/s':>10} {'LS-8 run(s)':>13}")
print(f"  {'-'*9} {'-'*6} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*13}")
for r in results:
    print(f"  {r['city']:<10} {r['N']:>6} {r['n_arcs']:>8} "
          f"{r['mean_ms']:>10.1f} {r['p95_ms']:>10.1f} "
          f"{r['evals_per_s']:>10.1f} {r['ls8_run_s']:>13.2f}")

# Scaling analysis
if len(results) >= 2:
    print("\n  Scaling (mean time vs N):")
    for i in range(1, len(results)):
        r0, r1 = results[i-1], results[i]
        n_ratio = r1["N"] / r0["N"]
        t_ratio = r1["mean_ms"] / r0["mean_ms"]
        exponent = np.log(t_ratio) / np.log(n_ratio) if n_ratio > 1 else 0.0
        print(f"    {r0['city']} -> {r1['city']}: N x{n_ratio:.2f}  time x{t_ratio:.2f}  "
              f"(approx O(N^{exponent:.2f}))")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    f"UE Evaluator Speed Benchmark  |  "
    f"max_range={args.ue_max_range}km  alpha_w={args.ue_alpha_wait}  "
    f"pd={args.ue_penalty_distance}  cap/s={args.ue_cap_per_server}",
    fontsize=10,
)

cities   = [r["city"] for r in results]
ns       = [r["N"]       for r in results]
means    = [r["mean_ms"] for r in results]
p95s     = [r["p95_ms"]  for r in results]
ls8_runs = [r["ls8_run_s"] for r in results]

# Panel 1: mean eval time vs N
ax = axes[0]
ax.bar(cities, means, color=["steelblue", "darkorange", "seagreen"], edgecolor="white", alpha=0.85)
for i, (city, m) in enumerate(zip(cities, means)):
    ax.text(i, m + 1, f"{m:.1f}ms", ha="center", fontsize=9)
ax.set_ylabel("Mean eval time (ms)")
ax.set_title("Mean UE eval time by instance size")

# Panel 2: scaling curve
ax = axes[1]
ax.plot(ns, means, "o-", color="steelblue", linewidth=2, markersize=8)
for city, n, m in zip(cities, ns, means):
    ax.annotate(f"{city}\nN={n}", (n, m), textcoords="offset points",
                xytext=(5, 5), fontsize=8)
ax.set_xlabel("N (number of nodes)")
ax.set_ylabel("Mean eval time (ms)")
ax.set_title("Scaling: eval time vs N")

# Panel 3: LS-8 total run time (8 UE evals = one LS pass)
ax = axes[2]
ax.bar(cities, ls8_runs, color=["steelblue", "darkorange", "seagreen"], edgecolor="white", alpha=0.85)
for i, (city, r) in enumerate(zip(cities, ls8_runs)):
    ax.text(i, r + 0.01, f"{r:.2f}s", ha="center", fontsize=9)
ax.set_ylabel("Time for 8-move UE LS run (s)")
ax.set_title("UE cost per full LS pass (8 moves)")
ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="0.5s threshold")
ax.legend(fontsize=8)

fig.tight_layout()
out_path = RESULTS_DIR / "ue_speed_benchmark.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")

# CSV summary
df_res = pd.DataFrame(results)
csv_out = RESULTS_DIR / "ue_speed_benchmark.csv"
df_res.to_csv(csv_out, index=False)
print(f"Data saved:  {csv_out}")
