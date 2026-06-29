"""
plot_monza_cp_vs_ue.py

Side-by-side map of Monza:
  Left  — Closest-Priority best solution
  Right — User Equilibrium best solution

Runs 300 ALNS iterations for each policy (same parameters as the logged runs
that produced the diagnostics/policy_comparison/ PNGs).

Output: results/diagnostics/policy_comparison/monza_cp_vs_ue.pdf
        results/diagnostics/policy_comparison/monza_cp_vs_ue.png

Usage:
    python scripts/plot_monza_cp_vs_ue.py
    python scripts/plot_monza_cp_vs_ue.py --max-iter 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evcs.geom import build_arcs
from evcs.destroy import destroy_multi_u
from evcs.greedy import reconstruct_u_dict_fast
from evcs.methods import greedy_schedule_multi_from_variants
from evcs.local_search import local_search_u_proxy
from evcs.ue_evaluator import UEEvaluator
from evcs.proxy import evaluate_u_numpy_greedy_binary

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--csv",         default="center_79_Monza_k400.csv")
p.add_argument("--seed",        type=int,   default=11)
p.add_argument("--T",           type=int,   default=6)
p.add_argument("--D",           type=float, default=2.0)
p.add_argument("--max-iter",    type=int,   default=300)
p.add_argument("--frac-remove", type=float, default=0.20)
p.add_argument("--ls-moves",    type=int,   default=8)
p.add_argument("--ls-frac",     type=float, default=0.08)
p.add_argument("--ue-alpha-wait",       type=float, default=10.0)
p.add_argument("--ue-noopt",            type=float, default=30.0)
p.add_argument("--ue-max-range",        type=float, default=2.0)
p.add_argument("--ue-penalty-distance", type=float, default=5.0)
p.add_argument("--ue-cap-per-server",   type=float, default=4.0)
args = p.parse_args()

DATA_DIR = PROJECT_ROOT / "data" / "input"
OUT_DIR  = PROJECT_ROOT / "results" / "diagnostics" / "policy_comparison"
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

pop        = np.maximum(df_raw["Aggregated_Population"].to_numpy(float), 0.0)
pop_share  = pop / pop.sum()
M          = coords_km.shape[0]

_period_budget = max(4, M // 30)   # = 13 for M=400
P_T = [_period_budget] * args.T

base     = pop_share * M
rng_inst = np.random.default_rng(args.seed)
demand_TM = np.zeros((args.T, M))
for t in range(args.T):
    sf = 1.0 + 0.2 * np.sin(2 * np.pi * t / args.T)
    demand_TM[t] = np.maximum(0.0, base * sf * (1.0 + rng_inst.normal(0, 0.05, M)))

distIJ, in_range, Ji, Ij = build_arcs(coords_km, coords_km, D=args.D, forbid_self=False)

Ij_int = {j: [] for j in range(M)}
Ji_int = {i: [] for i in range(M)}
for (i, j) in in_range:
    Ij_int[j].append(i)
    Ji_int[i].append(j)

J_i_list = [sorted(Ji_int[i], key=lambda j: distIJ[i, j]) for i in range(M)]

inst = {
    "coords_I":  coords_km,
    "coords_J":  coords_km.copy(),
    "demand_IT": demand_TM,
    "distIJ":    distIJ,
    "in_range":  in_range,
    "Ji":        Ji,
    "Ij":        Ij,
    "M":         M,
    "N":         M,
}

total_demand_t0 = float(demand_TM[0].sum())
print(f"Instance : {args.csv}")
print(f"N={M}  T={args.T}  D={args.D}km  |arcs|={len(in_range)}")
print(f"Budget   : {_period_budget}/period x{args.T} = {sum(P_T)} total chargers")


# ── UEEvaluator ───────────────────────────────────────────────────────────────
ue_mu = args.ue_cap_per_server + 1e-4
ue_eval = UEEvaluator(
    N=M, d=demand_TM[0], tau=distIJ, mu=ue_mu, s_max=MAX_CHARGERS,
    noopt_cost=args.ue_noopt, alpha_wait=args.ue_alpha_wait, N_bp=50,
    max_range=args.ue_max_range, penalty_distance=args.ue_penalty_distance,
)

def ue_fn(U):
    score, _ = ue_eval.evaluate(U, T=args.T, cumulative_install=CUMULATIVE)
    return float(score)


# ── Shared helpers ─────────────────────────────────────────────────────────────
def station_info(U_dict):
    chargers = np.array(
        [sum(int(U_dict.get((j, t), 0)) for t in range(args.T)) for j in range(M)],
        dtype=int,
    )
    open_mask = chargers > 0
    return np.where(open_mask)[0], chargers[open_mask]


def alns_loop(eval_fn, label, seed_offset=0):
    """Generic ALNS loop; eval_fn(U_dict) -> float."""
    rng = np.random.default_rng(args.seed + 1 + seed_offset)
    U_curr = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=args.seed)
    score_curr = eval_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    oi, cc = station_info(U_curr)
    print(f"\n[{label}] greedy: score={score_curr:.3f}  stations={len(oi)}  chargers={cc.sum()}")

    for it in range(1, args.max_iter + 1):
        base_sol = U_curr if rng.random() < 0.7 else U_best
        mode = str(rng.choice(DESTROY_MODES))

        U_partial, _ = destroy_multi_u(
            base_sol, inst, rng, P_T, args.frac_remove, mode,
            site_cap=MAX_CHARGERS, cumulative_install=CUMULATIVE,
        )
        U_recon, _ = reconstruct_u_dict_fast(
            U_partial, demand_TM, P_T, Ij_int,
            U_cap=MAX_CHARGERS, Q=Q_CAP, rng=rng,
            cumulative_install=CUMULATIVE,
        )
        visited, scores, _ = local_search_u_proxy(
            U_recon, inst, rng, P_T,
            demand_TM, J_i_list, distIJ,
            Q_CAP, args.T, M,
            CUMULATIVE, MAX_CHARGERS,
            args.ls_moves, args.ls_frac, LS_MODES,
            Ij_int=Ij_int, collect_visited=True, proxy_fn=eval_fn,
        )

        best_idx   = int(np.argmax(scores))
        iter_score = float(scores[best_idx])
        iter_U     = visited[best_idx]

        if iter_score >= score_curr:
            U_curr, score_curr = dict(iter_U), iter_score
        if iter_score > score_best:
            U_best, score_best = dict(iter_U), iter_score
            oi, cc = station_info(U_best)
            print(f"  iter {it:>3}  NEW BEST  score={score_best:.3f}  "
                  f"stations={len(oi)}  chargers={cc.sum()}")

    oi, cc = station_info(U_best)
    print(f"[{label}] done: best={score_best:.3f}  stations={len(oi)}  chargers={cc.sum()}")
    return U_best, score_best


# ── Run CP ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running Closest-Priority ALNS ...")

def cp_eval_fn(U):
    """Multi-period greedy binary coverage score (closest-priority proxy)."""
    return evaluate_u_numpy_greedy_binary(
        U, demand_TM=demand_TM, J_i_list=J_i_list, distIJ=distIJ,
        Q_cap=Q_CAP, T=args.T, N=M, cumulative_install=CUMULATIVE,
        pre_sorted_J_i=J_i_list,
    )

U_cp, score_cp = alns_loop(cp_eval_fn, "Closest-Priority", seed_offset=0)

# ── Run UE ────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running User-Equilibrium ALNS ...")
U_ue, score_ue = alns_loop(ue_fn, "User-Equilibrium", seed_offset=100)

ue_eval.dispose()

# ── Figure ────────────────────────────────────────────────────────────────────
CMAP      = plt.cm.coolwarm
CMAP_NORM = mcolors.Normalize(vmin=1, vmax=MAX_CHARGERS)

demand_t0   = demand_TM[0]
demand_norm = np.clip(demand_t0 / demand_t0.max(), 0, 1)
dot_sizes   = 6 + demand_norm * 28   # 6–34 pt²

# Use degree coordinates so axis labels read as lat/lon
coords_plot = coords_deg   # shape (M, 2): col0=lon, col1=lat

# ── FIX: compute correct lon/lat radii for a true circle on the map ──────────
# At latitude ~45.6°N: 1 deg lat ≈ 111 km, 1 deg lon ≈ 111 * cos(lat) km
mean_lat_rad = np.radians(coords_plot[:, 1].mean())
r_lat = args.D / 111.0
r_lon = args.D / (111.0 * np.cos(mean_lat_rad))
# ─────────────────────────────────────────────────────────────────────────────


def _draw(ax, U_dict, title):
    open_idx, cc = station_info(U_dict)

    # demand nodes
    ax.scatter(
        coords_plot[:, 0], coords_plot[:, 1],
        s=dot_sizes, c="#CCCCCC", linewidths=0,
        zorder=1, rasterized=True,
    )

    # coverage circles + station markers
    sm = None
    if len(open_idx) > 0:
        colors_rgba = CMAP(CMAP_NORM(cc))

        # ── FIX: use r_lon for width and r_lat for height ─────────────────
        for j, nc, rgba in zip(open_idx, cc, colors_rgba):
            circ = mpatches.Ellipse(
                xy=(coords_plot[j, 0], coords_plot[j, 1]),
                width=2 * r_lon, height=2 * r_lat,
                linewidth=0.7,
                edgecolor=rgba[:3],
                facecolor=(*rgba[:3], 0.07),
                zorder=2,
            )
            ax.add_patch(circ)
        # ─────────────────────────────────────────────────────────────────

        marker_sizes = np.clip(cc * 40, 60, 300)
        sc = ax.scatter(
            coords_plot[open_idx, 0], coords_plot[open_idx, 1],
            s=marker_sizes, c=cc, cmap=CMAP, norm=CMAP_NORM,
            edgecolors="black", linewidths=0.6,
            zorder=3,
        )
        sm = sc

        # charger count labels
        for j, nc in zip(open_idx, cc):
            ax.annotate(
                str(nc),
                xy=(coords_plot[j, 0], coords_plot[j, 1]),
                fontsize=6, ha="center", va="center",
                color="white", fontweight="bold", zorder=4,
            )

    ax.set_aspect(1.0 / np.cos(mean_lat_rad))
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude",  fontsize=9)
    ax.tick_params(labelsize=8)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f°"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f°"))
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    return sm, open_idx, cc


plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

oi_cp, cc_cp = station_info(U_cp)
oi_ue, cc_ue = station_info(U_ue)

title_cp = (f"Closest-Priority\n"
            f"{len(oi_cp)} stations,  {cc_cp.sum()} chargers")
title_ue = (f"User Equilibrium\n"
            f"{len(oi_ue)} stations,  {cc_ue.sum()} chargers")

fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
fig.suptitle("Monza — Closest-Priority vs. User Equilibrium",
             fontsize=12, fontweight="bold", y=1.01)

sm_cp, _, _ = _draw(axes[0], U_cp, title_cp)
sm_ue, _, _ = _draw(axes[1], U_ue, title_ue)

fig.subplots_adjust(left=0.06, right=0.84, top=0.93, bottom=0.09, wspace=0.28)

# shared colorbar on a fixed axes — not affected by tight_layout
sm_ref = sm_cp if sm_cp is not None else sm_ue
if sm_ref is not None:
    cax = fig.add_axes([0.87, 0.12, 0.018, 0.72])
    cbar = fig.colorbar(sm_ref, cax=cax, ticks=range(1, MAX_CHARGERS + 1))
    cbar.set_label("Chargers per station", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

legend_patches = [
    mpatches.Patch(facecolor="#CCCCCC",   label=f"Demand nodes  (N={M})"),
    mpatches.Patch(facecolor=CMAP(0.15),  label="Stations  (1 charger)"),
    mpatches.Patch(facecolor=CMAP(0.85),  label=f"Stations  ({MAX_CHARGERS} chargers)"),
    mpatches.Patch(facecolor="none", edgecolor="#888",
                   label=f"Coverage radius  {args.D} km"),
]
axes[0].legend(handles=legend_patches, loc="lower left", fontsize=7.5,
               framealpha=0.85, borderpad=0.6)

out_pdf = OUT_DIR / "monza_cp_vs_ue.pdf"
out_png = OUT_DIR / "monza_cp_vs_ue.png"
fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")
print(f"\nCP  -> {len(oi_cp)} stations, {cc_cp.sum()} chargers  (score {score_cp:.2f})")
print(f"UE  -> {len(oi_ue)} stations, {cc_ue.sum()} chargers  (score {score_ue:.2f})")
print("Done.")