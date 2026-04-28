"""
Quick diagnostic: generate N_SOLUTIONS diverse solutions, score each with
UEEvaluator (new params) and the greedy proxy, then plot both distributions.

Goal: UE std >> 0 across solutions (no saturation).

Usage:
    python scripts/diagnose_ue_spread.py
    python scripts/diagnose_ue_spread.py --ue-alpha-wait 50 --ue-penalty-distance 10
    python scripts/diagnose_ue_spread.py --ue-cap-per-server 1.0   # tighten capacity
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

from scipy.stats import spearmanr
from evcs.geom import build_arcs
from evcs.proxy import (
    evaluate_u_numpy_greedy_binary,
    compute_overflow_t0,
    evaluate_u_congestion_proxy,
)
from evcs.destroy import destroy_multi_u
from evcs.greedy import reconstruct_u_dict_fast
from evcs.methods import greedy_schedule_multi_from_variants
from evcs.ue_evaluator import UEEvaluator

# ── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--csv",               default="center_146_Verona_k250.csv")
p.add_argument("--seed",              type=int,   default=11)
p.add_argument("--D",                 type=float, default=2.0)
p.add_argument("--T",                 type=int,   default=6)
p.add_argument("--n-solutions",       type=int,   default=50)
p.add_argument("--ue-alpha-wait",     type=float, default=10.0)
p.add_argument("--ue-noopt",          type=float, default=30.0)
p.add_argument("--ue-max-range",      type=float, default=2.0)
p.add_argument("--ue-penalty-distance", type=float, default=5.0)
p.add_argument("--ue-cap-per-server", type=float, default=4.0)
args = p.parse_args()

DATA_DIR    = PROJECT_ROOT / "data" / "input"
RESULTS_DIR = PROJECT_ROOT / "results" / "diagnostics"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Instance ─────────────────────────────────────────────────────────────────
csv_path = DATA_DIR / args.csv
df_raw = pd.read_csv(csv_path)
coords_deg = df_raw[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float)
coords_km  = coords_deg.copy()
coords_km[:, 0] *= 111.0
coords_km[:, 1] *= 111.0

pop = np.maximum(df_raw["Aggregated_Population"].to_numpy(float), 0.0)
pop_share = pop / pop.sum()
M = coords_km.shape[0]
base = pop_share * M

rng_inst = np.random.default_rng(args.seed)
T = args.T
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

P_T = [8] * T
max_chargers = 6
Q_cap = 20.0
cumulative_install = True

print(f"Instance: {args.csv}  N={M}  T={T}  D={args.D}  |arcs|={len(in_range)}")
print(f"Total demand[t=0]: {demand_TM[0].sum():.2f}  (per node avg: {demand_TM[0].mean():.3f})")

# ── UEEvaluator ──────────────────────────────────────────────────────────────
ue_cap = args.ue_cap_per_server
ue_mu  = ue_cap + 1e-4

print(f"\nUEEvaluator params:")
print(f"  mu={ue_mu:.4f}  noopt_cost={args.ue_noopt}  max_range={args.ue_max_range} km")
print(f"  alpha_wait={args.ue_alpha_wait}  penalty_distance={args.ue_penalty_distance}")
print(f"  cap/server={ue_cap}  s_max={max_chargers}")
print(f"  Max total capacity: {P_T[0]} stations x {max_chargers} servers x {ue_cap:.1f} = "
      f"{P_T[0]*max_chargers*ue_cap:.1f}  vs total demand {demand_TM[0].sum():.1f}")

ue_eval = UEEvaluator(
    N=M, d=demand_TM[0], tau=distIJ, mu=ue_mu, s_max=max_chargers,
    noopt_cost=args.ue_noopt, alpha_wait=args.ue_alpha_wait, N_bp=50,
    max_range=args.ue_max_range, penalty_distance=args.ue_penalty_distance,
)

# ── Generate diverse solutions ────────────────────────────────────────────────
rng = np.random.default_rng(args.seed + 1)
destroy_modes = ["site_swap", "local_remove", "area_destroy"]

# Start from greedy
U_base = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=args.seed)

ue_scores       = []
bin_scores      = []   # binary proxy T=6/Q=20 (baseline)
overflow_scores = []   # period-0 congestion overflow (raw, unscaled)

print(f"\nEvaluating {args.n_solutions} solutions ...")
t0 = time.perf_counter()

for k in range(args.n_solutions):
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

    ue_score, _ = ue_eval.evaluate(U_sol, T=T, cumulative_install=cumulative_install)

    px_bin = evaluate_u_numpy_greedy_binary(
        U_sol, demand_TM=demand_TM, J_i_list=J_i_list, distIJ=distIJ,
        Q_cap=Q_cap, T=T, N=M, cumulative_install=cumulative_install,
    )
    overflow = compute_overflow_t0(U_sol, demand_TM[0], J_i_list, ue_mu, T, M)

    ue_scores.append(float(ue_score))
    bin_scores.append(float(px_bin))
    overflow_scores.append(float(overflow))

    if (k + 1) % 10 == 0:
        elapsed = time.perf_counter() - t0
        print(f"  {k+1:3d}/{args.n_solutions}  UE={ue_score:.3f}  "
              f"bin={px_bin:.1f}  overflow={overflow:.2f}  elapsed={elapsed:.1f}s")

ue_arr       = np.array(ue_scores)
bin_arr      = np.array(bin_scores)
overflow_arr = np.array(overflow_scores)


def _pearson(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or a[mask].std() < 1e-9 or b[mask].std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a[mask], b[mask])[0, 1])


def _spearman(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or a[mask].std() < 1e-9 or b[mask].std() < 1e-9:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


# ── Statistics ────────────────────────────────────────────────────────────────
sep = "=" * 60
print(f"\n{sep}")
print("UE score distribution:")
print(f"  min={ue_arr.min():.4f}  max={ue_arr.max():.4f}  "
      f"mean={ue_arr.mean():.4f}  std={ue_arr.std():.4f}")
print(f"  range={ue_arr.max()-ue_arr.min():.4f}   "
      f"std/mean={ue_arr.std()/max(ue_arr.mean(),1e-9):.4f}")
print(f"  unique values: {len(np.unique(ue_arr.round(4)))}")
saturation_flag = ue_arr.std() < 0.01 * ue_arr.mean()
print(f"  {'*** STILL SATURATED ***' if saturation_flag else '[OK -- spread detected]'}")

print(f"\nBinary proxy T={T}/Q={Q_cap}:")
print(f"  min={bin_arr.min():.4f}  max={bin_arr.max():.4f}  "
      f"mean={bin_arr.mean():.4f}  std={bin_arr.std():.4f}")
r_bin_p = _pearson(bin_arr, ue_arr)
r_bin_s = _spearman(bin_arr, ue_arr)
print(f"  r(Pearson)={r_bin_p:+.4f}   r(Spearman)={r_bin_s:+.4f}  vs UE")

print(f"\nOverflow (congestion term):")
print(f"  min={overflow_arr.min():.4f}  max={overflow_arr.max():.4f}  "
      f"mean={overflow_arr.mean():.4f}  std={overflow_arr.std():.4f}")

# ── Lambda grid search ────────────────────────────────────────────────────────
print(f"\n{sep}")
print("Lambda grid search: congestion_proxy = binary - lam * overflow")

overflow_scale = overflow_arr.std()
if overflow_scale < 1e-9:
    print("  WARNING: overflow is constant -- congestion term adds nothing.")
    best_lam = 0.0
    best_r   = r_bin_p
    lam_grid = np.array([0.0])
    r_grid   = np.array([r_bin_p])
else:
    # search up to 3x (bin std / overflow std) so the term can dominate
    lam_max  = 3.0 * bin_arr.std() / overflow_scale
    lam_grid = np.linspace(0.0, lam_max, 2001)
    r_grid   = np.array([
        _pearson(bin_arr - lam * overflow_arr, ue_arr)
        for lam in lam_grid
    ])
    finite_mask = np.isfinite(r_grid)
    if finite_mask.any():
        best_idx = int(np.argmax(r_grid[finite_mask]))
        best_lam = float(lam_grid[finite_mask][best_idx])
        best_r   = float(r_grid[finite_mask][best_idx])
    else:
        best_lam = 0.0
        best_r   = r_bin_p

    best_s = _spearman(bin_arr - best_lam * overflow_arr, ue_arr)
    print(f"\n  Optimal lambda={best_lam:.4f}")
    print(f"  Congestion proxy:  r(Pearson)={best_r:+.4f}   r(Spearman)={best_s:+.4f}")
    print(f"  Baseline binary:   r(Pearson)={r_bin_p:+.4f}   r(Spearman)={r_bin_s:+.4f}")
    gain = best_r - r_bin_p
    print(f"  Gain: {gain:+.4f}  ({'improvement' if gain > 0 else 'no improvement'})")
    print(f"\n  --> Use --lambda-cong {best_lam:.4f} in collect_trajectories.py")

cong_best = bin_arr - best_lam * overflow_arr

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle(
    f"{args.csv}  N={M}  |  UE: alpha_w={args.ue_alpha_wait}, "
    f"pd={args.ue_penalty_distance}, range={args.ue_max_range}km, cap/s={ue_cap}",
    fontsize=10,
)


def _hist(ax, arr, color, xlabel, title):
    bins = 20 if arr.std() > 1e-9 else 1
    ax.hist(arr, bins=bins, color=color, edgecolor="white")
    ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"mu={arr.mean():.2f} sigma={arr.std():.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(fontsize=8)


def _scatter(ax, x, y, xlabel, ylabel, title, color):
    rp = _pearson(x, y)
    rs = _spearman(x, y)
    ax.scatter(x, y, alpha=0.6, s=30, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nPearson r={rp:.3f}  Spearman rho={rs:.3f}")
    if x.std() > 1e-9:
        m = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(m, xs), color="red", linewidth=1.2, linestyle="--")


_hist(axes[0, 0], ue_arr,       "steelblue",  "UE score (demand covered)",    "UE full-eval distribution")
_hist(axes[0, 1], bin_arr,      "darkorange", f"Binary proxy T={T}/Q={Q_cap}", f"Binary proxy T={T}/Q={Q_cap}")
_hist(axes[0, 2], overflow_arr, "tomato",     "Overflow (congestion)",         "Overflow distribution")

_scatter(axes[1, 0], bin_arr, ue_arr,
         f"Binary T={T}/Q={Q_cap}", "UE score",
         f"Baseline binary vs UE  (r={r_bin_p:.3f})", "darkorange")

_scatter(axes[1, 1], cong_best, ue_arr,
         f"binary - {best_lam:.3f}*overflow", "UE score",
         f"Best congestion proxy vs UE  (r={best_r:.3f})", "seagreen")

# r vs lambda curve
ax = axes[1, 2]
if len(lam_grid) > 1:
    ax.plot(lam_grid, r_grid, color="navy", linewidth=1.5)
    ax.axvline(best_lam, color="red", linestyle="--", linewidth=1.2,
               label=f"best lam={best_lam:.3f}  r={best_r:.3f}")
    ax.axhline(r_bin_p, color="darkorange", linestyle=":", linewidth=1.2,
               label=f"baseline r={r_bin_p:.3f}")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Pearson r  (congestion proxy vs UE)")
    ax.set_title("r vs lambda  (grid search)")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "overflow constant\n(no lambda search)", ha="center", va="center",
            transform=ax.transAxes)
    ax.set_title("r vs lambda")

fig.tight_layout()
out_path = RESULTS_DIR / "ue_spread_diagnostic.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")
