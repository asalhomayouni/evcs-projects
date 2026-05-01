"""
plot_alns_ue.py

Full ALNS loop (mirrors run_DR_multi exactly) with UEEvaluator as both the
full evaluator and the local-search proxy.

Saves:
  - A map PNG every time U_best improves (incumbent_XX_iter{it}_ue{score:.1f}.png)
  - A detailed best-solution map at the end
  - An objective-trajectory plot (current vs best UE score per iteration)

Usage:
    python scripts/plot_alns_ue.py
    python scripts/plot_alns_ue.py --csv center_102_Vicenza_k125.csv --max-iter 30
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
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evcs.geom import build_arcs
from evcs.destroy import destroy_multi_u
from evcs.greedy import reconstruct_u_dict_fast
from evcs.methods import greedy_schedule_multi_from_variants
from evcs.local_search import local_search_u_proxy
from evcs.ue_evaluator import UEEvaluator

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--csv",           default="center_146_Verona_k250.csv")
p.add_argument("--max-iter",      type=int,   default=25)
p.add_argument("--seed",          type=int,   default=11)
p.add_argument("--T",             type=int,   default=6)
p.add_argument("--D",             type=float, default=2.0)
p.add_argument("--frac-remove",   type=float, default=0.20)
p.add_argument("--ls-moves",      type=int,   default=8)
p.add_argument("--ls-frac",       type=float, default=0.08)
p.add_argument("--accept-eps",    type=float, default=0.0,
               help="Accept neighbour if score >= best - eps (exploration window)")
p.add_argument("--time-limit",    type=float, default=float("inf"),
               help="Stop after this many seconds (default: no limit)")
p.add_argument("--ue-alpha-wait",       type=float, default=10.0)
p.add_argument("--ue-noopt",            type=float, default=10.0)
p.add_argument("--ue-max-range",        type=float, default=2.0)
p.add_argument("--ue-penalty-distance", type=float, default=5.0)
p.add_argument("--ue-cap-per-server",   type=float, default=4.0)
p.add_argument("--tag", default="",
               help="Optional suffix appended to output directory name")
p.add_argument("--results-base", default="results/diagnostics/alns_ue",
               help="Base output directory relative to project root (default: results/diagnostics/alns_ue)")
p.add_argument("--traj-width",   type=float, default=9.0,
               help="Trajectory plot figure width in inches (default 9)")
p.add_argument("--marker-size",  type=int,   default=80,
               help="Size of improvement dots on trajectory plot (default 80)")
args = p.parse_args()

DATA_DIR = PROJECT_ROOT / "data" / "input"

instance_stem = Path(args.csv).stem
dir_name = instance_stem + (f"_{args.tag}" if args.tag else "")
OUT_DIR = PROJECT_ROOT / args.results_base / dir_name
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHARGERS  = 6
Q_CAP         = 20.0
CUMULATIVE    = True
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]
LS_MODES      = ("site_swap", "local_remove")

# ── Load instance ─────────────────────────────────────────────────────────────
df_raw     = pd.read_csv(DATA_DIR / args.csv)
coords_deg = df_raw[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float)
coords_km  = coords_deg * 111.0

pop       = np.maximum(df_raw["Aggregated_Population"].to_numpy(float), 0.0)
pop_share = pop / pop.sum()
M         = coords_km.shape[0]

# Budget scales with instance size.  Rule: total capacity = (N//30)*T*mu ≈ 0.8N
# which keeps total capacity below total demand, making the LP UB informative.
_period_budget = max(4, M // 30)
P_T            = [_period_budget] * args.T

base      = pop_share * M

rng_inst  = np.random.default_rng(args.seed)
demand_TM = np.zeros((args.T, M))
for t in range(args.T):
    sf = 1.0 + 0.2 * np.sin(2 * np.pi * t / args.T)
    demand_TM[t] = np.maximum(
        0.0, base * sf * (1.0 + rng_inst.normal(0, 0.05, M))
    )

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

total_demand_t0 = float(demand_TM[0].sum())
print(f"Instance : {args.csv}")
print(f"N={M}  T={args.T}  D={args.D}km  |arcs|={len(in_range)}")
print(f"Budget   : {_period_budget}/period  x{args.T} periods  = {sum(P_T)} total chargers")
print(f"Total demand (t=0): {total_demand_t0:.2f}")

# ── UEEvaluator ───────────────────────────────────────────────────────────────
ue_mu = args.ue_cap_per_server + 1e-4
print(
    f"\nUEEvaluator  mu={ue_mu:.4f}  max_range={args.ue_max_range}km  "
    f"alpha_wait={args.ue_alpha_wait}  penalty_dist={args.ue_penalty_distance}"
)
ue_eval = UEEvaluator(
    N=M, d=demand_TM[0], tau=distIJ, mu=ue_mu, s_max=MAX_CHARGERS,
    noopt_cost=args.ue_noopt, alpha_wait=args.ue_alpha_wait, N_bp=50,
    max_range=args.ue_max_range, penalty_distance=args.ue_penalty_distance,
)

def ue_fn(U):
    score, _ = ue_eval.evaluate(U, T=args.T, cumulative_install=CUMULATIVE)
    return float(score)

# ── LP relaxation upper bound ─────────────────────────────────────────────────
# Tight LP relaxation: relaxes server integrality but uses the exact Erlang-C
# capacity ceiling from ue_eval._cap and bounds flow per station by both the
# continuous server allocation AND the hard per-station max-server capacity.
# Budget = sum(P_T), consistent with cumulative install at the final period.
try:
    import gurobipy as _gp
    from gurobipy import GRB as _GRB
    from evcs.model_grb import _get_env as _get_env_ub
    _bgt  = float(sum(P_T))         # total charger budget (scales with N now)
    _d0   = demand_TM[0]
    # Use UEEvaluator's exact per-station max capacity (mu*s_max - eps)
    _cap_max = np.array([float(ue_eval._cap[j, MAX_CHARGERS - 1]) for j in range(M)])

    _ri_ub = {i: [] for i in range(M)}
    _rj_ub = {j: [] for j in range(M)}
    for (_i, _j) in in_range:
        _ri_ub[_i].append(_j)
        _rj_ub[_j].append(_i)

    _gm = _gp.Model(env=_get_env_ub()); _gm.setParam("OutputFlag", 0)
    _y   = _gm.addVars(in_range, lb=0.0)
    _yn  = _gm.addVars(M,        lb=0.0)
    _sv  = _gm.addVars(M,        lb=0.0, ub=float(MAX_CHARGERS))
    _lam = _gm.addVars(M,        lb=0.0)
    _gm.update()

    _gm.setObjective(_gp.quicksum(_lam[j] for j in range(M)), _GRB.MAXIMIZE)

    for _i in range(M):
        _gm.addConstr(_gp.quicksum(_y[_i, _j] for _j in _ri_ub[_i]) + _yn[_i] == float(_d0[_i]))
    for _j in range(M):
        if _rj_ub[_j]:
            _gm.addConstr(_lam[_j] == _gp.quicksum(_y[_i, _j] for _i in _rj_ub[_j]))
            # Bound (1): continuous server allocation (main LP relaxation)
            _gm.addConstr(_lam[_j] <= _sv[_j] * ue_mu)
            # Bound (2): hard cap at exact max-server Erlang capacity
            _gm.addConstr(_lam[_j] <= float(_cap_max[_j]))
        else:
            _gm.addConstr(_lam[_j] == 0.0)
            _gm.addConstr(_sv[_j]  == 0.0)

    # Global budget from new dynamic P_T
    _gm.addConstr(_gp.quicksum(_sv[_j] for _j in range(M)) <= _bgt)

    _gm.update(); _gm.optimize()
    lp_ub = float(_gm.ObjVal) if _gm.SolCount > 0 else float(_d0.sum())
    _gm.dispose()
    print(f"LP relaxation upper bound: {lp_ub:.3f}  (budget={_bgt:.0f}  cap/station={_cap_max[0]:.2f})")
except Exception as _exc:
    lp_ub = None
    print(f"LP UB skipped: {_exc}")

# ── Plotting helpers ──────────────────────────────────────────────────────────
CMAP      = plt.cm.coolwarm
CMAP_NORM = mcolors.Normalize(vmin=1, vmax=MAX_CHARGERS)

demand_t0   = demand_TM[0]
demand_norm = np.clip(demand_t0 / demand_t0.max(), 0, 1)
dot_sizes   = 4 + demand_norm * 30


def station_info(U_dict):
    chargers = np.array(
        [sum(int(U_dict.get((j, t), 0)) for t in range(args.T)) for j in range(M)],
        dtype=int,
    )
    open_mask = chargers > 0
    return np.where(open_mask)[0], chargers[open_mask]


def draw_map(ax, U_dict, title_main, title_sub="", label_chargers=False):
    open_idx, cc = station_info(U_dict)

    ax.scatter(
        coords_km[:, 0], coords_km[:, 1],
        s=dot_sizes, c="lightgrey", linewidths=0,
        zorder=1, rasterized=True,
    )

    sm = None
    if len(open_idx) > 0:
        colors_rgba = CMAP(CMAP_NORM(cc))
        for j, nc, rgba in zip(open_idx, cc, colors_rgba):
            ax.add_patch(mpatches.Circle(
                xy=(coords_km[j, 0], coords_km[j, 1]),
                radius=args.D,
                linewidth=0.8,
                edgecolor=rgba[:3],
                facecolor=(*rgba[:3], 0.07),
                zorder=2,
            ))
        sc = ax.scatter(
            coords_km[open_idx, 0], coords_km[open_idx, 1],
            s=np.clip(cc * 35, 55, 280),
            c=cc, cmap=CMAP, norm=CMAP_NORM,
            edgecolors="black", linewidths=0.6,
            zorder=3,
        )
        sm = sc
        if label_chargers:
            for j, nc in zip(open_idx, cc):
                ax.annotate(str(nc), xy=(coords_km[j, 0], coords_km[j, 1]),
                            fontsize=6.5, ha="center", va="center",
                            color="white", fontweight="bold", zorder=4)

    ax.set_aspect("equal")
    ax.set_xlabel("x  [km]", fontsize=8)
    ax.set_ylabel("y  [km]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title_main, fontsize=9, pad=4)
    if title_sub:
        ax.text(0.5, -0.09, title_sub, transform=ax.transAxes,
                ha="center", fontsize=8, color="#444")
    return sm


LEGEND_PATCHES = [
    mpatches.Patch(facecolor="lightgrey",  label=f"Demand nodes  (N={M})"),
    mpatches.Patch(facecolor=CMAP(0.15),   label="Stations  (few chargers)"),
    mpatches.Patch(facecolor=CMAP(0.85),   label="Stations  (many chargers)"),
    mpatches.Patch(facecolor="none", edgecolor="grey", linestyle="--",
                   label=f"Coverage  r={args.D}km"),
]


def save_map(U_dict, fname, title_main, title_sub="",
             label_chargers=False, figsize=(9, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    sm = draw_map(ax, U_dict, title_main, title_sub, label_chargers)
    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
        cbar.set_label("Chargers / station", fontsize=8)
        cbar.set_ticks(range(1, MAX_CHARGERS + 1))
        cbar.ax.tick_params(labelsize=7)
    ax.legend(handles=LEGEND_PATCHES, loc="upper right", fontsize=7.5, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── ALNS loop (mirrors run_DR_multi exactly, UE replaces full_eval_from_U) ───
rng = np.random.default_rng(args.seed + 1)

print("\nInitial greedy solution ...")
U_curr = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=args.seed)
score_curr = ue_fn(U_curr)
U_best     = dict(U_curr)
score_best = score_curr

open_idx0, cc0 = station_info(U_curr)
print(f"  Greedy UE={score_curr:.3f}  stations={len(open_idx0)}  chargers={cc0.sum()}")

ue_str0 = f"{score_curr:.1f}".replace(".", "p")
save_map(
    U_curr,
    fname=f"best_00_iter000_ue{ue_str0}.png",
    title_main=f"New best #0  (greedy, iter 0)  |  UE={score_curr:.2f}  "
               f"|  {len(open_idx0)} stations  |  {cc0.sum()} chargers",
    title_sub=f"Served: {score_curr:.1f} / {total_demand_t0:.1f}  "
              f"({100*score_curr/total_demand_t0:.1f}%)",
)
print(f"  Saved: best_00_iter000_ue{ue_str0}.png")

# Trajectory storage
traj_iter        = [0]
traj_score_curr  = [score_curr]
traj_score_best  = [score_best]
traj_mode        = ["init"]
traj_accepted    = [True]

# Incumbent (new-best) history for summary
incumbents  = [(0, score_best, "init")]
step_idx    = 1   # counts every accepted step (including downhill)

print(f"\nALNS  max_iter={args.max_iter}  frac_remove={args.frac_remove}  "
      f"ls_moves={args.ls_moves}  accept_eps={args.accept_eps}\n")
header = f"{'iter':>5}  {'mode':<16}  {'iter_UE':>9}  {'curr_UE':>9}  {'best_UE':>9}  {'accept':>7}  {'new_best':>9}"
print(header)
print("-" * len(header))

t_alns = time.perf_counter()

for it in range(1, args.max_iter + 1):
    if time.perf_counter() - t_alns >= args.time_limit:
        print(f"  [time limit {args.time_limit:.0f}s reached at iter {it}]")
        break


    # ── Destroy ──────────────────────────────────────────────────────────────
    base = U_curr if rng.random() < 0.7 else U_best
    mode = str(rng.choice(DESTROY_MODES))

    U_partial, _ = destroy_multi_u(
        base, inst, rng, P_T, args.frac_remove, mode,
        site_cap=MAX_CHARGERS, cumulative_install=CUMULATIVE,
    )

    # ── Reconstruct ──────────────────────────────────────────────────────────
    U_recon, _ = reconstruct_u_dict_fast(
        U_partial, demand_TM, P_T, Ij_int,
        U_cap=MAX_CHARGERS, Q=Q_CAP, rng=rng,
        cumulative_install=CUMULATIVE,
    )

    # ── UE-guided local search ────────────────────────────────────────────────
    # proxy_fn=ue_fn means every LS step is evaluated by UEEvaluator directly.
    # collect_visited=True returns all visited solutions so we can pick the
    # UE-best across the full LS trajectory (same logic as run_DR_multi batch eval).
    visited, ue_scores, _ = local_search_u_proxy(
        U_recon, inst, rng, P_T,
        demand_TM, J_i_list, distIJ,
        Q_CAP, args.T, M,
        CUMULATIVE, MAX_CHARGERS,
        args.ls_moves, args.ls_frac, LS_MODES,
        Ij_int=Ij_int, collect_visited=True, proxy_fn=ue_fn,
    )

    # Best UE score across all LS-visited solutions (mirrors batch eval in DR)
    best_iter_idx   = int(np.argmax(ue_scores))
    best_iter_score = float(ue_scores[best_iter_idx])
    best_iter_U     = visited[best_iter_idx]

    # ── Two-tier acceptance (same as run_DR_multi) ────────────────────────────
    accepted = best_iter_score >= score_curr - args.accept_eps
    new_best = best_iter_score > score_best

    if accepted:
        U_curr     = dict(best_iter_U)
        score_curr = best_iter_score

    if new_best:
        U_best     = dict(best_iter_U)
        score_best = best_iter_score
        incumbents.append((it, score_best, mode))
        # Save a map only when the best score is beaten
        open_i, cc_i = station_info(U_best)
        ue_str = f"{score_best:.1f}".replace(".", "p")
        fname  = f"best_{step_idx:02d}_iter{it:03d}_ue{ue_str}.png"
        save_map(
            U_best, fname,
            title_main=(f"New best #{step_idx}  (iter {it}, {mode})  |  "
                        f"UE={score_best:.2f}  |  "
                        f"{len(open_i)} stations  |  {cc_i.sum()} chargers"),
            title_sub=(f"Served: {score_best:.1f} / {total_demand_t0:.1f}  "
                       f"({100*score_best/total_demand_t0:.1f}%)  |  "
                       f"destroy={mode}"),
        )
        step_idx += 1

    traj_iter.append(it)
    traj_score_curr.append(score_curr)
    traj_score_best.append(score_best)
    traj_mode.append(mode)
    traj_accepted.append(accepted)

    print(f"{it:>5}  {mode:<16}  {best_iter_score:>9.3f}  {score_curr:>9.3f}  "
          f"{score_best:>9.3f}  {'YES' if accepted else 'no':>7}  "
          f"{'** NEW BEST **' if new_best else '':>9}")

elapsed = time.perf_counter() - t_alns
print(f"\nALNS done in {elapsed:.1f}s  ({elapsed/args.max_iter*1000:.0f}ms/iter avg)")
print(f"Best UE={score_best:.3f}  new-bests found={len(incumbents)-1}  accepted steps={step_idx-1}")

# ── Best solution detailed map ────────────────────────────────────────────────
open_b, cc_b = station_info(U_best)
pct_b = 100.0 * score_best / total_demand_t0
save_map(
    U_best,
    fname="best_solution.png",
    title_main=(f"BEST  |  UE={score_best:.2f}  |  "
                f"{len(open_b)} stations  |  {cc_b.sum()} chargers"),
    title_sub=(f"Served demand: {score_best:.1f} / {total_demand_t0:.1f}  ({pct_b:.1f}%)  "
               f"|  charger count on each marker"),
    label_chargers=True,
    figsize=(11, 10),
)
print("Saved: best_solution.png")

# ── Objective trajectory plot ─────────────────────────────────────────────────
best_scores = [inc[1] for inc in incumbents]
n_jumps     = len(incumbents) - 1

# Conference-quality style
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.9,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
})

fig, ax = plt.subplots(figsize=(args.traj_width, 6))
ax.grid(axis="y", alpha=0.20, linewidth=0.5)

# ── Best-score staircase ──────────────────────────────────────────────────────
ax.plot(traj_iter, traj_score_best,
        color="#C0392B", linewidth=2.2, zorder=3,
        solid_capstyle="round", label="ALNS best")

# ── Improvement dots (no drop lines, no annotations) ─────────────────────────
imp_iters  = [inc[0] for inc in incumbents[1:]]
imp_scores = [inc[1] for inc in incumbents[1:]]
ax.scatter(imp_iters, imp_scores,
           color="#C0392B", s=args.marker_size, zorder=5,
           edgecolors="white", linewidths=1.0)

# ── Greedy baseline ───────────────────────────────────────────────────────────
ax.axhline(
    best_scores[0],
    color="#2471A3", linewidth=1.4, linestyle="--", alpha=0.85, zorder=2,
    label=f"Greedy  {best_scores[0]:.1f}",
)

# ── LP relaxation upper bound ─────────────────────────────────────────────────
_y_lo   = best_scores[0]
_ref_hi = lp_ub if lp_ub is not None else best_scores[-1]
_jspan  = max(_ref_hi - _y_lo, 1e-6)
_pad    = max(_jspan * 0.06, 0.05)
_ylim_lo = _y_lo - _pad
_ylim_hi = _ref_hi + _pad

if lp_ub is not None:
    ax.axhline(
        lp_ub,
        color="#1E8449", linewidth=1.8, linestyle=(0, (5, 3)),
        alpha=1.0, zorder=4,
        label=f"LP UB  {lp_ub:.1f}",
    )

ax.set_xlabel("Iteration", fontsize=11)
ax.set_ylabel("UE score", fontsize=11)
ax.set_xlim(0, traj_iter[-1])
ax.set_ylim(_ylim_lo, _ylim_hi)

ax.set_title(
    f"{Path(args.csv).stem}   $N={M}$,  $T={args.T}$,  budget/period $={_period_budget}$",
    fontsize=11, pad=8,
)
ax.legend(fontsize=9, frameon=False, loc="lower right")

fig.tight_layout()
traj_fname = "alns_trajectory.png"
fig.savefig(OUT_DIR / traj_fname, dpi=180, bbox_inches="tight")
plt.close(fig)
plt.rcdefaults()
print(f"Saved: {traj_fname}")

# ── Incumbent summary ─────────────────────────────────────────────────────────
print(f"\nIncumbent history:")
print(f"  {'#':<4} {'iter':>6}  {'UE score':>10}  {'destroy mode'}")
print(f"  {'-'*3} {'-'*6}  {'-'*10}  {'-'*16}")
for k, (inc_it, inc_score, inc_mode) in enumerate(incumbents):
    delta = f"+{inc_score - incumbents[k-1][1]:.3f}" if k > 0 else "baseline"
    print(f"  {k:<4} {inc_it:>6}  {inc_score:>10.3f}  {inc_mode:<16}  ({delta})")

print(f"\nAll plots -> {OUT_DIR}")
ue_eval.dispose()
print("Done.")
