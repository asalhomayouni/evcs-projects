"""
plot_dr_solutions.py

Generates feasible solutions (greedy base + destroy-reconstruct variants),
evaluates each with UEEvaluator, and saves scatter-map plots for visual
sanity checking of solution quality and geographic plausibility.

What to look for:
  - Stations near demand clusters, not scattered randomly
  - Coverage circles overlapping with demand nodes
  - Higher-charger stations in denser demand areas
  - UE scores vary across solutions (no saturation)

Usage:
    python scripts/plot_dr_solutions.py
    python scripts/plot_dr_solutions.py --n-solutions 15 --seed 42
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
from evcs.ue_evaluator import UEEvaluator

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--csv",           default="center_146_Verona_k250.csv")
p.add_argument("--n-solutions",   type=int,   default=10)
p.add_argument("--seed",          type=int,   default=11)
p.add_argument("--T",             type=int,   default=6)
p.add_argument("--D",             type=float, default=2.0)
p.add_argument("--ue-alpha-wait",       type=float, default=10.0)
p.add_argument("--ue-noopt",            type=float, default=30.0)
p.add_argument("--ue-max-range",        type=float, default=2.0)
p.add_argument("--ue-penalty-distance", type=float, default=5.0)
p.add_argument("--ue-cap-per-server",   type=float, default=4.0)
args = p.parse_args()

DATA_DIR = PROJECT_ROOT / "data" / "input"
OUT_DIR  = PROJECT_ROOT / "results" / "diagnostics" / "solution_maps"
OUT_DIR.mkdir(parents=True, exist_ok=True)

P_T          = [8] * args.T
MAX_CHARGERS = 6
Q_CAP        = 20.0
CUMULATIVE   = True
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]

# ── Load instance ─────────────────────────────────────────────────────────────
csv_path   = DATA_DIR / args.csv
df_raw     = pd.read_csv(csv_path)

coords_deg = df_raw[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float)
# Algorithm works in km (uniform 111 km/degree approximation, consistent with
# how build_arcs is called everywhere else in this project).
coords_km  = coords_deg * 111.0

pop        = np.maximum(df_raw["Aggregated_Population"].to_numpy(float), 0.0)
pop_share  = pop / pop.sum()
M          = coords_km.shape[0]
base       = pop_share * M

rng_inst   = np.random.default_rng(args.seed)
demand_TM  = np.zeros((args.T, M))
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

inst = {
    "coords_I":  coords_km,
    "coords_J":  coords_km.copy(),
    "demand_IT": demand_TM,
    "distIJ":    distIJ,
    "in_range":  in_range,
    "Ji": Ji, "Ij": Ij,
    "M": M, "N": M,
}

total_demand_t0 = float(demand_TM[0].sum())
print(f"Instance : {args.csv}")
print(f"N={M}  T={args.T}  D={args.D}km  |arcs|={len(in_range)}")
print(f"Total demand (t=0): {total_demand_t0:.2f}")

# ── UEEvaluator ───────────────────────────────────────────────────────────────
ue_mu = args.ue_cap_per_server + 1e-4
print(
    f"\nInitialising UEEvaluator  "
    f"mu={ue_mu:.4f}  max_range={args.ue_max_range}km  "
    f"alpha_wait={args.ue_alpha_wait}  penalty_dist={args.ue_penalty_distance}"
)
t_build = time.perf_counter()
ue_eval = UEEvaluator(
    N=M, d=demand_TM[0], tau=distIJ, mu=ue_mu, s_max=MAX_CHARGERS,
    noopt_cost=args.ue_noopt, alpha_wait=args.ue_alpha_wait, N_bp=50,
    max_range=args.ue_max_range, penalty_distance=args.ue_penalty_distance,
)
print(f"  Built in {time.perf_counter() - t_build:.2f}s")

# ── Generate solutions ────────────────────────────────────────────────────────
rng    = np.random.default_rng(args.seed + 1)
U_base = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=args.seed)

solutions = []  # list of (label, U_dict, ue_score)

# Solution 0: greedy base
t0 = time.perf_counter()
score0, _ = ue_eval.evaluate(U_base, T=args.T, cumulative_install=CUMULATIVE)
solutions.append(("greedy", U_base, float(score0)))
print(f"\n  [0] greedy    UE={score0:.3f}  ({(time.perf_counter()-t0)*1000:.0f}ms)")

# Solutions 1..N-1: destroy-reconstruct variants
for k in range(args.n_solutions - 1):
    frac = float(rng.uniform(0.3, 0.6))
    mode = str(rng.choice(DESTROY_MODES))

    U_partial, _ = destroy_multi_u(
        U_base, inst, rng, P_T, frac, mode,
        site_cap=MAX_CHARGERS, cumulative_install=CUMULATIVE,
    )
    U_sol, _ = reconstruct_u_dict_fast(
        U_partial, demand_TM, P_T, Ij_int,
        U_cap=MAX_CHARGERS, Q=Q_CAP, rng=rng,
        cumulative_install=CUMULATIVE,
    )
    t0 = time.perf_counter()
    score, _ = ue_eval.evaluate(U_sol, T=args.T, cumulative_install=CUMULATIVE)
    solutions.append((f"dr_{k+1:02d}", U_sol, float(score)))
    print(
        f"  [{k+1:2d}] dr_{k+1:02d}   UE={score:.3f}  "
        f"mode={mode:<14}  frac={frac:.2f}  "
        f"({(time.perf_counter()-t0)*1000:.0f}ms)"
    )

# ── Summary ───────────────────────────────────────────────────────────────────
def station_info(U_dict):
    """Returns arrays: open_indices, cumulative_charger_counts."""
    chargers = np.array(
        [sum(int(U_dict.get((j, t), 0)) for t in range(args.T)) for j in range(M)],
        dtype=int,
    )
    open_mask = chargers > 0
    return np.where(open_mask)[0], chargers[open_mask]

scores = [s for _, _, s in solutions]
best_k = int(np.argmax(scores))

print("\n" + "="*65)
print(f"{'Idx':<5} {'Label':<12} {'UE score':>10} {'Stations':>9} {'Chargers':>9}")
print("-"*65)
for k, (label, U_sol, score) in enumerate(solutions):
    open_idx, cc = station_info(U_sol)
    marker = " <-- BEST" if k == best_k else ""
    print(f"{k:<5} {label:<12} {score:>10.3f} {len(open_idx):>9} {cc.sum():>9}{marker}")
print("="*65)

ue_scores = np.array(scores)
print(f"\nUE score stats:  mean={ue_scores.mean():.3f}  std={ue_scores.std():.3f}  "
      f"range={ue_scores.max()-ue_scores.min():.3f}")

# ── Plotting setup ────────────────────────────────────────────────────────────
# Colourmap: blue = few chargers, red = many
CMAP      = plt.cm.coolwarm
CMAP_NORM = mcolors.Normalize(vmin=1, vmax=MAX_CHARGERS)

# Demand node sizes scaled by demand magnitude
demand_t0   = demand_TM[0]
demand_norm = np.clip(demand_t0 / demand_t0.max(), 0, 1)
dot_sizes   = 4 + demand_norm * 30   # 4 to 34 pt²


def _draw_solution(ax, U_dict, title_main, title_sub="", label_chargers=False):
    open_idx, cc = station_info(U_dict)

    # Demand nodes (grey, sized by demand)
    ax.scatter(
        coords_km[:, 0], coords_km[:, 1],
        s=dot_sizes, c="lightgrey", linewidths=0,
        zorder=1, rasterized=True,
    )

    sm = None
    if len(open_idx) > 0:
        colors_rgba = CMAP(CMAP_NORM(cc))

        # Coverage circles
        for j, nc, rgba in zip(open_idx, cc, colors_rgba):
            ax.add_patch(mpatches.Circle(
                xy=(coords_km[j, 0], coords_km[j, 1]),
                radius=args.D,
                linewidth=0.8,
                edgecolor=rgba[:3],
                facecolor=(*rgba[:3], 0.07),
                zorder=2,
            ))

        # Station markers (size proportional to charger count)
        marker_sizes = np.clip(cc * 35, 55, 280)
        sc = ax.scatter(
            coords_km[open_idx, 0], coords_km[open_idx, 1],
            s=marker_sizes, c=cc, cmap=CMAP, norm=CMAP_NORM,
            edgecolors="black", linewidths=0.6,
            zorder=3,
        )
        sm = sc

        if label_chargers:
            for j, nc in zip(open_idx, cc):
                ax.annotate(
                    str(nc),
                    xy=(coords_km[j, 0], coords_km[j, 1]),
                    fontsize=6.5, ha="center", va="center",
                    color="white", fontweight="bold", zorder=4,
                )

    ax.set_aspect("equal")
    ax.set_xlabel("x  [km  (lon × 111)]", fontsize=8)
    ax.set_ylabel("y  [km  (lat × 111)]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title_main, fontsize=9, pad=4)
    if title_sub:
        ax.text(
            0.5, -0.09, title_sub,
            transform=ax.transAxes, ha="center", fontsize=8, color="#444",
        )
    return sm


# ── Per-solution maps ─────────────────────────────────────────────────────────
print(f"\nSaving {len(solutions)} iteration maps -> {OUT_DIR}")

legend_patches = [
    mpatches.Patch(facecolor="lightgrey",     label=f"Demand nodes  (N={M})"),
    mpatches.Patch(facecolor=CMAP(0.15),      label="Stations  (few chargers)"),
    mpatches.Patch(facecolor=CMAP(0.85),      label="Stations  (many chargers)"),
    mpatches.Patch(facecolor="none",
                   edgecolor="grey", linestyle="--",
                   label=f"Coverage radius  r={args.D}km"),
]

for k, (label, U_sol, score) in enumerate(solutions):
    open_idx, cc = station_info(U_sol)
    pct = 100.0 * score / total_demand_t0

    title_main = (
        f"Iter {k}  ({label})  |  UE = {score:.2f}  "
        f"|  {len(open_idx)} stations  |  {cc.sum()} chargers"
    )
    title_sub = f"Served demand: {score:.1f} / {total_demand_t0:.1f}  ({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(9, 8))
    sm = _draw_solution(ax, U_sol, title_main, title_sub)

    if sm is not None:
        cbar = fig.colorbar(sm, ax=ax, fraction=0.030, pad=0.02)
        cbar.set_label("Chargers / station", fontsize=8)
        cbar.set_ticks(range(1, MAX_CHARGERS + 1))
        cbar.ax.tick_params(labelsize=7)

    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=7.5, framealpha=0.85)

    score_str = f"{score:.1f}".replace(".", "p")
    fname     = f"iter_{k:02d}_ue_{score_str}.png"
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")

# ── Best-solution detailed map ────────────────────────────────────────────────
best_label, best_U, best_score = solutions[best_k]
open_idx_b, cc_b = station_info(best_U)
pct_b = 100.0 * best_score / total_demand_t0

title_main = (
    f"BEST  (iter {best_k}, {best_label})  |  "
    f"UE = {best_score:.2f}  |  "
    f"{len(open_idx_b)} stations  |  {cc_b.sum()} chargers"
)
title_sub = (
    f"Demand served: {best_score:.1f} / {total_demand_t0:.1f}  ({pct_b:.1f}%)   "
    f"|   charger labels shown on markers"
)

fig, ax = plt.subplots(figsize=(11, 10))
sm = _draw_solution(ax, best_U, title_main, title_sub, label_chargers=True)

if sm is not None:
    cbar = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Chargers / station", fontsize=8)
    cbar.set_ticks(range(1, MAX_CHARGERS + 1))
    cbar.ax.tick_params(labelsize=7)

best_patches = [
    mpatches.Patch(facecolor="lightgrey",
                   label=f"Demand nodes  (N={M}),  size ∝ demand"),
    mpatches.Patch(facecolor=CMAP(0.15), label="Stations  (1 charger)"),
    mpatches.Patch(facecolor=CMAP(0.85), label=f"Stations  ({MAX_CHARGERS} chargers)"),
    mpatches.Patch(facecolor="none",
                   edgecolor="grey", linestyle="--",
                   label=f"Coverage radius  r={args.D}km"),
]
ax.legend(handles=best_patches, loc="upper right", fontsize=8, framealpha=0.9)

fig.tight_layout()
best_fname = "best_solution.png"
fig.savefig(OUT_DIR / best_fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  {best_fname}  (detailed best)")

# ── UE score distribution panel ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(
    f"{args.csv}  N={M}  T={args.T}  D={args.D}km  |  "
    f"alpha_w={args.ue_alpha_wait}  range={args.ue_max_range}km  "
    f"pd={args.ue_penalty_distance}  cap/s={args.ue_cap_per_server}",
    fontsize=9,
)

ax = axes[0]
colors_bar = ["gold" if k == best_k else "steelblue" for k in range(len(solutions))]
ax.bar(range(len(solutions)), ue_scores, color=colors_bar, edgecolor="white", linewidth=0.5)
ax.axhline(ue_scores.mean(), color="red", linestyle="--", linewidth=1.2,
           label=f"mean={ue_scores.mean():.2f}")
ax.set_xlabel("Solution index")
ax.set_ylabel("UE score")
ax.set_title("UE scores across solutions  (gold = best)")
ax.legend(fontsize=8)
ax.tick_params(labelsize=8)

ax = axes[1]
n_open_list    = []
n_charger_list = []
for _, U_sol, _ in solutions:
    oi, cc = station_info(U_sol)
    n_open_list.append(len(oi))
    n_charger_list.append(int(cc.sum()))

ax.scatter(n_open_list, ue_scores, s=60, c="steelblue", edgecolors="black",
           linewidths=0.5, zorder=3)
for k, (no, sc) in enumerate(zip(n_open_list, ue_scores)):
    ax.annotate(str(k), (no, sc), fontsize=6.5, ha="left", va="bottom",
                xytext=(2, 2), textcoords="offset points")
ax.set_xlabel("Number of open stations")
ax.set_ylabel("UE score")
ax.set_title("UE score vs open stations")
ax.tick_params(labelsize=8)

fig.tight_layout()
summary_fname = "solution_summary.png"
fig.savefig(OUT_DIR / summary_fname, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  {summary_fname}  (score summary)")

ue_eval.dispose()
print(f"\nAll plots saved to:  {OUT_DIR}")
print("Done.")
