"""
Proof of concept: UE-guided vs binary-proxy-guided local search.

For N_TRIALS random destroy-reconstruct candidates (same starting base):
  - Run LS guided by binary proxy (production setting, fast)
  - Run LS guided by UE full-eval (experimental, ~0.024s/step on N=250)
  - Post-hoc: evaluate UE for every solution visited during binary LS
  - Compare UE score trajectories and final UE scores

Usage:
    python scripts/benchmark_ue_ls.py
    python scripts/benchmark_ue_ls.py --n-trials 50 --ls-moves 10
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
from evcs.proxy import evaluate_u_numpy_greedy_binary
from evcs.destroy import destroy_multi_u
from evcs.greedy import reconstruct_u_dict_fast
from evcs.methods import greedy_schedule_multi_from_variants
from evcs.local_search import local_search_u_proxy
from evcs.ue_evaluator import UEEvaluator

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--csv",                 default="center_146_Verona_k250.csv")
p.add_argument("--seed",                type=int,   default=11)
p.add_argument("--D",                   type=float, default=2.0)
p.add_argument("--T",                   type=int,   default=6)
p.add_argument("--n-trials",            type=int,   default=30,
               help="Number of independent D-R candidates to test (default 30)")
p.add_argument("--ls-moves",            type=int,   default=8,
               help="Max LS moves per trial (early stop still active; default 8)")
p.add_argument("--ue-alpha-wait",       type=float, default=10.0)
p.add_argument("--ue-noopt",            type=float, default=30.0)
p.add_argument("--ue-max-range",        type=float, default=2.0)
p.add_argument("--ue-penalty-distance", type=float, default=5.0)
p.add_argument("--ue-cap-per-server",   type=float, default=4.0)
args = p.parse_args()

DATA_DIR    = PROJECT_ROOT / "data" / "input"
RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Instance ──────────────────────────────────────────────────────────────────
csv_path = DATA_DIR / args.csv
df_raw   = pd.read_csv(csv_path)

coords_deg = df_raw[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float)
coords_km  = coords_deg.copy()
coords_km[:, 0] *= 111.0
coords_km[:, 1] *= 111.0

pop       = np.maximum(df_raw["Aggregated_Population"].to_numpy(float), 0.0)
pop_share = pop / pop.sum()
M         = coords_km.shape[0]
base      = pop_share * M

rng_inst = np.random.default_rng(args.seed)
T        = args.T
demand_TM = np.zeros((T, M))
for t in range(T):
    sf = 1.0 + 0.2 * np.sin(2 * np.pi * t / T)
    demand_TM[t] = np.maximum(0.0, base * sf * (1.0 + rng_inst.normal(0, 0.05, M)))

distIJ, in_range, Ji, Ij = build_arcs(coords_km, coords_km, D=args.D, forbid_self=False)

Ij_int = {j: [] for j in range(M)}
Ji_int = {i: [] for i in range(M)}
for (i, j) in in_range:
    Ij_int[j].append(i)
    Ji_int[i].append(j)

J_i_list = [sorted(Ji_int[i], key=lambda j: distIJ[i, j]) for i in range(M)]

inst = {
    "coords_I": coords_km, "coords_J": coords_km.copy(),
    "demand_IT": demand_TM, "distIJ": distIJ,
    "in_range": in_range, "Ji": Ji, "Ij": Ij, "M": M, "N": M,
}

P_T               = [8] * T
max_chargers      = 6
Q_cap             = 20.0
cumulative_install = True
ls_modes          = ("site_swap", "local_remove")
ls_frac_remove    = 0.08
destroy_modes     = ["site_swap", "local_remove", "area_destroy"]
frac_remove       = 0.3

print(f"Instance: {args.csv}  N={M}  T={T}  D={args.D}  |arcs|={len(in_range)}")

# ── UEEvaluator ───────────────────────────────────────────────────────────────
ue_cap = args.ue_cap_per_server
ue_mu  = ue_cap + 1e-4

print(f"\nInitialising UEEvaluator (mu={ue_mu:.4f}, max_range={args.ue_max_range} km, "
      f"alpha_wait={args.ue_alpha_wait}, penalty_distance={args.ue_penalty_distance}) ...")

ue_eval = UEEvaluator(
    N=M, d=demand_TM[0], tau=distIJ, mu=ue_mu, s_max=max_chargers,
    noopt_cost=args.ue_noopt, alpha_wait=args.ue_alpha_wait, N_bp=50,
    max_range=args.ue_max_range, penalty_distance=args.ue_penalty_distance,
)

def ue_fn(U):
    score, _ = ue_eval.evaluate(U, T=T, cumulative_install=cumulative_install)
    return float(score)

def bin_fn(U):
    return float(evaluate_u_numpy_greedy_binary(
        U, demand_TM=demand_TM, J_i_list=J_i_list, distIJ=distIJ,
        Q_cap=Q_cap, T=T, N=M, cumulative_install=cumulative_install,
    ))

# ── Greedy base ───────────────────────────────────────────────────────────────
rng    = np.random.default_rng(args.seed + 1)
U_base = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=args.seed)

# ── Trials ────────────────────────────────────────────────────────────────────
# For each trial:
#   1. Destroy-reconstruct from U_base -> U_start (same for both methods)
#   2. Binary LS: collect all visited solutions, then batch-evaluate UE post-hoc
#   3. UE LS:     UE scores come directly from the LS (UE is the proxy_fn)
#   4. Record final UE score and UE trajectory for each method

print(f"\nRunning {args.n_trials} trials  (binary LS + UE LS each) ...")
print("  [B] = binary LS pass,  [U] = UE LS pass\n")

# Storage for per-trial results
bin_final_ue  = []   # final UE score after binary LS
ue_final_ue   = []   # final UE score after UE LS

# UE trajectories: list of arrays (one per trial, variable length due to early stop)
bin_traj_ue  = []   # UE scores of each visited solution during binary LS
ue_traj_ue   = []   # UE scores of each visited solution during UE LS (= proxy scores)

bin_steps     = []
ue_steps      = []

t0 = time.perf_counter()

for k in range(args.n_trials):
    # Same D-R candidate for both methods
    frac = rng.uniform(0.3, 0.6)
    mode = str(rng.choice(destroy_modes))

    U_partial, _ = destroy_multi_u(
        U_base, inst, rng, P_T, frac, mode,
        site_cap=max_chargers, cumulative_install=cumulative_install,
    )
    U_start, _ = reconstruct_u_dict_fast(
        U_partial, demand_TM, P_T, Ij_int,
        U_cap=max_chargers, Q=Q_cap, rng=rng,
        cumulative_install=cumulative_install,
    )

    # ── Binary LS ─────────────────────────────────────────────────────────────
    rng_b = np.random.default_rng(args.seed + 1000 + k)
    visited_b, _, _ = local_search_u_proxy(
        U_start, inst, rng_b, P_T,
        demand_TM, J_i_list, distIJ,
        Q_cap, T, M,
        cumulative_install, max_chargers,
        args.ls_moves, ls_frac_remove, ls_modes,
        Ij_int=Ij_int, collect_visited=True, proxy_fn=None,
    )
    # Post-hoc UE eval for every visited solution
    ue_b = np.array([ue_fn(U) for U in visited_b])
    bin_traj_ue.append(ue_b)
    bin_final_ue.append(float(ue_b.max()))   # best UE seen during binary LS
    bin_steps.append(len(visited_b) - 1)

    # ── UE LS ─────────────────────────────────────────────────────────────────
    rng_u = np.random.default_rng(args.seed + 1000 + k)   # same rng seed for fair comparison
    visited_u, ue_scores_u, _ = local_search_u_proxy(
        U_start, inst, rng_u, P_T,
        demand_TM, J_i_list, distIJ,
        Q_cap, T, M,
        cumulative_install, max_chargers,
        args.ls_moves, ls_frac_remove, ls_modes,
        Ij_int=Ij_int, collect_visited=True, proxy_fn=ue_fn,
    )
    ue_u = np.array(ue_scores_u, dtype=float)
    ue_traj_ue.append(ue_u)
    ue_final_ue.append(float(ue_u.max()))
    ue_steps.append(len(visited_u) - 1)

    if (k + 1) % 5 == 0:
        elapsed = time.perf_counter() - t0
        print(f"  trial {k+1:3d}/{args.n_trials}  "
              f"[B] final_ue={bin_final_ue[-1]:.3f} steps={bin_steps[-1]}  "
              f"[U] final_ue={ue_final_ue[-1]:.3f} steps={ue_steps[-1]}  "
              f"elapsed={elapsed:.1f}s")

elapsed_total = time.perf_counter() - t0

# ── Statistics ────────────────────────────────────────────────────────────────
bin_arr = np.array(bin_final_ue)
ue_arr  = np.array(ue_final_ue)
delta   = ue_arr - bin_arr

sep = "=" * 60
print(f"\n{sep}")
print(f"Results over {args.n_trials} trials  (elapsed: {elapsed_total:.1f}s)")
print(f"\nFinal UE score (max UE seen across LS trajectory):")
print(f"  Binary LS:  mean={bin_arr.mean():.3f}  std={bin_arr.std():.3f}  "
      f"min={bin_arr.min():.3f}  max={bin_arr.max():.3f}")
print(f"  UE LS:      mean={ue_arr.mean():.3f}   std={ue_arr.std():.3f}  "
      f"min={ue_arr.min():.3f}  max={ue_arr.max():.3f}")
print(f"\nDelta (UE_LS - Binary_LS) per trial:")
print(f"  mean={delta.mean():+.3f}  std={delta.std():.3f}  "
      f"min={delta.min():+.3f}  max={delta.max():+.3f}")
frac_better = float((delta > 1e-6).mean())
print(f"  UE LS beats binary LS in {frac_better*100:.0f}% of trials")

print(f"\nLS steps taken (early stop):")
print(f"  Binary LS:  mean={np.mean(bin_steps):.2f}  max={max(bin_steps)}")
print(f"  UE LS:      mean={np.mean(ue_steps):.2f}   max={max(ue_steps)}")

# ── Build trajectory arrays aligned to step index ─────────────────────────────
# Pad shorter trajectories with NaN for plotting
max_steps = max(max(len(t) for t in bin_traj_ue),
                max(len(t) for t in ue_traj_ue))

def _pad(trajs, length):
    arr = np.full((len(trajs), length), np.nan)
    for i, t in enumerate(trajs):
        arr[i, :len(t)] = t
    return arr

bin_mat = _pad(bin_traj_ue, max_steps)
ue_mat  = _pad(ue_traj_ue,  max_steps)

steps = np.arange(max_steps)

def _mean_std(mat):
    mn  = np.nanmean(mat, axis=0)
    std = np.nanstd(mat, axis=0)
    n   = np.sum(~np.isnan(mat), axis=0)
    return mn, std, n

bin_mn, bin_std, bin_n = _mean_std(bin_mat)
ue_mn,  ue_std,  ue_n  = _mean_std(ue_mat)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"{args.csv}  N={M}  |  {args.n_trials} trials, max {args.ls_moves} LS moves\n"
    f"UE: alpha_w={args.ue_alpha_wait}, range={args.ue_max_range}km, "
    f"pd={args.ue_penalty_distance}, cap/s={ue_cap}",
    fontsize=9,
)

# Panel 1: mean UE trajectory +/- 1 std
ax = axes[0]
ax.plot(steps, bin_mn, color="darkorange", linewidth=2, label="Binary LS (proxy)")
ax.fill_between(steps,
                np.clip(bin_mn - bin_std, 0, None),
                bin_mn + bin_std,
                color="darkorange", alpha=0.25)
ax.plot(steps, ue_mn, color="seagreen", linewidth=2, label="UE LS (oracle)")
ax.fill_between(steps,
                np.clip(ue_mn - ue_std, 0, None),
                ue_mn + ue_std,
                color="seagreen", alpha=0.25)
ax.set_xlabel("LS step")
ax.set_ylabel("UE score")
ax.set_title("Mean UE trajectory (+/-1 std)")
ax.legend(fontsize=9)
ax.set_xlim(0, max_steps - 1)

# Panel 2: box plot of final UE scores
ax = axes[1]
bp = ax.boxplot([bin_arr, ue_arr], labels=["Binary LS", "UE LS"],
                patch_artist=True, medianprops={"color": "black", "linewidth": 2})
bp["boxes"][0].set_facecolor("darkorange")
bp["boxes"][1].set_facecolor("seagreen")
for patch in bp["boxes"]:
    patch.set_alpha(0.7)
ax.set_ylabel("Final UE score (max over trajectory)")
ax.set_title(f"Final UE score comparison\n"
             f"UE LS better in {frac_better*100:.0f}% of trials  |  "
             f"mean delta={delta.mean():+.2f}")

# Panel 3: per-trial scatter (binary final vs UE final)
ax = axes[2]
lo = min(bin_arr.min(), ue_arr.min()) - 1
hi = max(bin_arr.max(), ue_arr.max()) + 1
ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, alpha=0.5, label="y=x (tie)")
ax.scatter(bin_arr, ue_arr, alpha=0.65, s=40, color="steelblue")
ax.set_xlabel("Binary LS final UE score")
ax.set_ylabel("UE LS final UE score")
ax.set_title("Per-trial: UE LS vs Binary LS\n(above diagonal = UE LS wins)")
ax.legend(fontsize=9)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)

fig.tight_layout()
out_path = RESULTS_DIR / "ue_ls_comparison.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")
