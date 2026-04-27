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
from evcs.proxy import evaluate_u_ue_greedy, evaluate_u_numpy_greedy_binary
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
print(f"  Max total capacity: {P_T[0]} stations × {max_chargers} servers × {ue_cap:.1f} = "
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

ue_scores   = []
proxy_scores = []
bin_scores   = []

print(f"\nEvaluating {args.n_solutions} solutions ...")
t0 = time.perf_counter()

for k in range(args.n_solutions):
    # Heavy destroy → reconstruct to get a diverse solution
    frac = rng.uniform(0.3, 0.7)   # randomise how much to destroy
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

    # old multi-period proxy (T=6, Q_cap=20) — kept for comparison
    px_bin_t6 = evaluate_u_numpy_greedy_binary(
        U_sol, demand_TM=demand_TM, J_i_list=J_i_list, distIJ=distIJ,
        Q_cap=Q_cap, T=T, N=M, cumulative_install=cumulative_install,
    )
    # scope-matched proxy: collapse cumulative servers into period 0, evaluate period-0 demand.
    # This gives capacity = s_j_cumulative * ue_cap (same as UE LP) and demand = demand_TM[0].
    U_collapsed = {(j, 0): sum(int(U_sol.get((j, t_), 0)) for t_ in range(T))
                   for j in range(M) if any(U_sol.get((j, t_), 0) for t_ in range(T))}
    px_bin_t1 = evaluate_u_numpy_greedy_binary(
        U_collapsed, demand_TM=demand_TM[0:1], J_i_list=J_i_list, distIJ=distIJ,
        Q_cap=ue_cap, T=1, N=M, cumulative_install=False,
    )

    ue_scores.append(float(ue_score))
    proxy_scores.append(float(px_bin_t6))
    bin_scores.append(float(px_bin_t1))

    if (k + 1) % 10 == 0:
        elapsed = time.perf_counter() - t0
        print(f"  {k+1:3d}/{args.n_solutions}  UE={ue_score:.3f}  "
              f"bin_t6={px_bin_t6:.1f}  bin_t1={px_bin_t1:.3f}  elapsed={elapsed:.1f}s")

ue_arr    = np.array(ue_scores)
proxy_arr = np.array(proxy_scores)
bin_arr   = np.array(bin_scores)

def _corr(a, b, label):
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or a[mask].std() < 1e-9 or b[mask].std() < 1e-9:
        return float("nan"), float("nan")
    r_p = float(np.corrcoef(a[mask], b[mask])[0, 1])
    r_s = float(spearmanr(a[mask], b[mask]).statistic)
    print(f"  r(Pearson)={r_p:+.4f}   r(Spearman)={r_s:+.4f}   [{label}]")
    return r_p, r_s

# ── Statistics ────────────────────────────────────────────────────────────────
sep = "="*60
print(f"\n{sep}")
print("UE score distribution:")
print(f"  min={ue_arr.min():.4f}  max={ue_arr.max():.4f}  "
      f"mean={ue_arr.mean():.4f}  std={ue_arr.std():.4f}")
print(f"  range={ue_arr.max()-ue_arr.min():.4f}   std/mean={ue_arr.std()/max(ue_arr.mean(),1e-9):.4f}")
print(f"  unique values: {len(np.unique(ue_arr.round(4)))}")

saturation_flag = ue_arr.std() < 0.01 * ue_arr.mean()
print(f"\n  {'*** STILL SATURATED ***' if saturation_flag else '[OK — spread detected]'}")

print(f"\nBinary proxy T=6/Q={Q_cap} (old, multi-period):")
print(f"  min={proxy_arr.min():.4f}  max={proxy_arr.max():.4f}  "
      f"mean={proxy_arr.mean():.4f}  std={proxy_arr.std():.4f}")
sat_px = proxy_arr.std() < 0.01 * proxy_arr.mean()
print(f"  {'*** SATURATED ***' if sat_px else '[OK]'}")

print(f"\nBinary proxy T=1/Q={ue_cap} (scope-matched to UE):")
print(f"  min={bin_arr.min():.4f}  max={bin_arr.max():.4f}  "
      f"mean={bin_arr.mean():.4f}  std={bin_arr.std():.4f}")
sat_bin = bin_arr.std() < 0.01 * bin_arr.mean()
print(f"  {'*** SATURATED ***' if sat_bin else '[OK]'}")

print(f"\nCorrelation with UE full-eval:")
r_ue_p,  r_ue_s  = _corr(proxy_arr, ue_arr, f"binary T=6/Q={Q_cap}  vs UE")
r_bin_p, r_bin_s = _corr(bin_arr,   ue_arr, f"binary T=1/Q={ue_cap} vs UE  [scope-matched]")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    f"{args.csv}  N={M}  |  UE: α_w={args.ue_alpha_wait}, pd={args.ue_penalty_distance}, "
    f"range={args.ue_max_range}km, cap/s={ue_cap}",
    fontsize=10,
)

def _hist(ax, arr, color, xlabel, title):
    bins = 20 if arr.std() > 1e-9 else 1
    ax.hist(arr, bins=bins, color=color, edgecolor="white")
    ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=1.5,
               label=f"μ={arr.mean():.2f} σ={arr.std():.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend(fontsize=8)

def _scatter(ax, x, y, r_p, r_s, xlabel, ylabel, title, color):
    ax.scatter(x, y, alpha=0.6, s=30, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nPearson r={r_p:.3f}  Spearman ρ={r_s:.3f}")
    if x.std() > 1e-9:
        m = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.polyval(m, xs), color="red", linewidth=1.2, linestyle="--")

_hist(axes[0, 0], ue_arr,    "steelblue",   "UE score (λ covered)",         "UE full-eval")
_hist(axes[0, 1], proxy_arr, "darkorange",  f"Binary proxy T=6/Q={Q_cap}",   f"Binary T=6/Q={Q_cap}  (old)")
_hist(axes[0, 2], bin_arr,   "mediumpurple", f"Binary proxy T=1/Q={ue_cap}", f"Binary T=1/Q={ue_cap}  (scope-matched)")

r_ue_s2  = r_ue_s  if np.isfinite(r_ue_s)  else 0.0
r_bin_s2 = r_bin_s if np.isfinite(r_bin_s) else 0.0
_scatter(axes[1, 0], proxy_arr, ue_arr, r_ue_p,  r_ue_s2,
         f"Binary T=6/Q={Q_cap}", "UE score",
         f"Binary T=6/Q={Q_cap} vs UE", "darkorange")
_scatter(axes[1, 1], bin_arr,   ue_arr, r_bin_p, r_bin_s2,
         f"Binary T=1/Q={ue_cap}", "UE score",
         f"Binary T=1/Q={ue_cap} vs UE  ← scope-matched", "mediumpurple")

ax = axes[1, 2]
ax.scatter(bin_arr, proxy_arr, alpha=0.6, s=30, color="teal")
if bin_arr.std() > 1e-9 and proxy_arr.std() > 1e-9:
    r_pp = float(np.corrcoef(bin_arr, proxy_arr)[0, 1])
    m = np.polyfit(bin_arr, proxy_arr, 1)
    xs = np.linspace(bin_arr.min(), bin_arr.max(), 100)
    ax.plot(xs, np.polyval(m, xs), color="red", linewidth=1.2, linestyle="--")
    ax.set_title(f"T=1/Q={ue_cap} vs T=6/Q={Q_cap}\nr={r_pp:.3f}")
else:
    ax.set_title("T=1 vs T=6 proxy\n(one is constant)")
ax.set_xlabel(f"Binary T=1/Q={ue_cap}")
ax.set_ylabel(f"Binary T=6/Q={Q_cap}")

fig.tight_layout()
out_path = RESULTS_DIR / "ue_spread_diagnostic.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved: {out_path}")
