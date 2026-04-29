"""
plot_ue_param_sweep.py

Parameter sweep over UE model parameters (alpha_wait, noopt_cost, penalty_distance).
For each combination a short D-R search (no LS) finds a representative solution
optimised under those parameters, then the layout is plotted.

Two 3×3 panel grids are produced:
  Grid 1 — alpha_wait × noopt_cost     (penalty_distance fixed)
  Grid 2 — alpha_wait × penalty_dist   (noopt_cost fixed)

Plus individual high-res maps for every combination.

Usage:
    python scripts/plot_ue_param_sweep.py
    python scripts/plot_ue_param_sweep.py --csv center_102_Vicenza_k125.csv
"""

import sys
import time
from itertools import product
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

# ── Config ────────────────────────────────────────────────────────────────────
CSV_FILE   = "center_146_Verona_k250.csv"
SEED       = 11
T          = 6
D          = 2.0
P_T        = [8] * T
MAX_CHARGERS = 6
Q_CAP        = 20.0
CUMULATIVE   = True
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]
N_DR_ITER    = 20      # D-R steps per combo (no LS — fast survey)
UE_CAP       = 4.0    # cap per server (fixed throughout)
UE_MU        = UE_CAP + 1e-4
UE_MAX_RANGE = 2.0

# ── Parameter grids ───────────────────────────────────────────────────────────
ALPHA_WAIT_VALS   = [1.0,  10.0,  25.0]   # rows
NOOPT_VALS        = [5.0,  10.0,  30.0]   # cols  (Grid 1)
PENALTY_DIST_VALS = [1.0,   5.0,  10.0]   # cols  (Grid 2)

# Fixed values when the other axis is being swept
FIX_PENALTY_DIST = 5.0   # used in Grid 1
FIX_NOOPT        = 10.0  # used in Grid 2

# "Current" (tuned) parameter set — highlighted with a gold border in each panel
CURRENT = dict(alpha_wait=10.0, noopt=10.0, pd=5.0)

# ── Output ────────────────────────────────────────────────────────────────────
OUT_DIR = (PROJECT_ROOT / "results" / "diagnostics"
           / "param_sweep" / Path(CSV_FILE).stem)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load instance ─────────────────────────────────────────────────────────────
df_raw     = pd.read_csv(PROJECT_ROOT / "data" / "input" / CSV_FILE)
coords_deg = df_raw[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float)
coords_km  = coords_deg * 111.0

pop        = np.maximum(df_raw["Aggregated_Population"].to_numpy(float), 0.0)
pop_share  = pop / pop.sum()
M          = coords_km.shape[0]
base       = pop_share * M

rng_inst   = np.random.default_rng(SEED)
demand_TM  = np.zeros((T, M))
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

total_demand = float(demand_TM[0].sum())
print(f"Instance : {CSV_FILE}  N={M}  T={T}  D={D}km")
print(f"Total demand (t=0): {total_demand:.2f}")
print(f"D-R iters per combo: {N_DR_ITER}")

# ── Greedy base (same for all combos, seeded identically) ────────────────────
U_greedy = greedy_schedule_multi_from_variants(inst, P_T, D, seed=SEED)

# ── Core: solve under one parameter set ──────────────────────────────────────
def solve_combo(alpha_wait, noopt, penalty_dist, label):
    """Quick D-R search under given UE params. Returns (U_best, score_best, n_open, n_char)."""
    ue = UEEvaluator(
        N=M, d=demand_TM[0], tau=distIJ, mu=UE_MU, s_max=MAX_CHARGERS,
        noopt_cost=noopt, alpha_wait=alpha_wait, N_bp=50,
        max_range=UE_MAX_RANGE, penalty_distance=penalty_dist,
    )
    ue_fn = lambda U: float(ue.evaluate(U, T=T, cumulative_install=CUMULATIVE)[0])

    rng      = np.random.default_rng(SEED + 1)
    U_curr   = dict(U_greedy)
    s_curr   = ue_fn(U_curr)
    U_best   = dict(U_curr)
    s_best   = s_curr
    n_improv = 0

    for _ in range(N_DR_ITER):
        mode = str(rng.choice(DESTROY_MODES))
        U_part, _ = destroy_multi_u(
            U_curr, inst, rng, P_T, 0.40, mode,
            site_cap=MAX_CHARGERS, cumulative_install=CUMULATIVE,
        )
        U_recon, _ = reconstruct_u_dict_fast(
            U_part, demand_TM, P_T, Ij_int,
            U_cap=MAX_CHARGERS, Q=Q_CAP, rng=rng, cumulative_install=CUMULATIVE,
        )
        sc = ue_fn(U_recon)
        if sc > s_best:
            U_best, s_best, n_improv = dict(U_recon), sc, n_improv + 1
        if sc >= s_curr:          # strict-improvement current accept
            U_curr, s_curr = dict(U_recon), sc

    chargers = np.array(
        [sum(int(U_best.get((j, t), 0)) for t in range(T)) for j in range(M)], dtype=int
    )
    n_open = int((chargers > 0).sum())
    n_char = int(chargers.sum())
    pct    = 100.0 * s_best / total_demand

    print(f"  {label:<35}  UE={s_best:7.3f}  ({pct:5.1f}%)  "
          f"stations={n_open}  chargers={n_char}  improv={n_improv}")
    ue.dispose()
    return U_best, s_best, n_open, n_char, chargers

# ── Plotting helpers ──────────────────────────────────────────────────────────
CMAP      = plt.cm.coolwarm
CMAP_NORM = mcolors.Normalize(vmin=1, vmax=MAX_CHARGERS)

demand_norm = np.clip(demand_TM[0] / demand_TM[0].max(), 0, 1)
dot_sizes   = 3 + demand_norm * 22


def draw_solution(ax, chargers, ue_score, n_open, pct, show_circles=True):
    """Draw demand dots + station markers + optional coverage circles onto ax."""
    open_idx = np.where(chargers > 0)[0]
    cc       = chargers[open_idx]

    ax.scatter(coords_km[:, 0], coords_km[:, 1],
               s=dot_sizes, c="lightgrey", linewidths=0, zorder=1, rasterized=True)

    if len(open_idx) > 0 and show_circles:
        rgba_arr = CMAP(CMAP_NORM(cc))
        for j, nc, rgba in zip(open_idx, cc, rgba_arr):
            ax.add_patch(mpatches.Circle(
                (coords_km[j, 0], coords_km[j, 1]), radius=D,
                linewidth=0.6, edgecolor=rgba[:3],
                facecolor=(*rgba[:3], 0.06), zorder=2,
            ))
        ax.scatter(coords_km[open_idx, 0], coords_km[open_idx, 1],
                   s=np.clip(cc * 28, 45, 220),
                   c=cc, cmap=CMAP, norm=CMAP_NORM,
                   edgecolors="black", linewidths=0.4, zorder=3)

    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False,
                   labelleft=False, labelbottom=False)


def save_individual(chargers, ue_score, n_open, n_char, pct, fname, title):
    fig, ax = plt.subplots(figsize=(8, 7))
    draw_solution(ax, chargers, ue_score, n_open, pct)
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=CMAP_NORM)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.028, pad=0.02)
    cb.set_label("Chargers / station", fontsize=8)
    cb.set_ticks(range(1, MAX_CHARGERS + 1))
    cb.ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=8.5, pad=5)
    ax.set_xlabel("x [km]", fontsize=8)
    ax.set_ylabel("y [km]", fontsize=8)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True,
                   labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_panel(grid_results, row_vals, col_vals, row_label, col_label,
               fixed_label, fname, highlight_fn):
    """
    Draw a len(row_vals) × len(col_vals) panel of solution maps.
    grid_results: dict[(row_val, col_val)] -> (chargers, ue_score, n_open, n_char, pct)
    highlight_fn: (row_val, col_val) -> bool  (True = gold border = current params)
    """
    nrow, ncol = len(row_vals), len(col_vals)
    fig, axes  = plt.subplots(nrow, ncol, figsize=(ncol * 4.8, nrow * 4.2))
    fig.suptitle(
        f"{CSV_FILE}  N={M}  D={D}km  |  {N_DR_ITER} D-R iters per combo\n"
        f"Fixed: {fixed_label}   |   Sweep: {row_label} (rows)  ×  {col_label} (cols)\n"
        f"Gold border = current tuned parameters",
        fontsize=9, y=1.01,
    )

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=CMAP_NORM)
    sm.set_array([])

    for ri, rv in enumerate(row_vals):
        for ci, cv in enumerate(col_vals):
            ax = axes[ri, ci]
            chargers, ue_score, n_open, n_char, pct = grid_results[(rv, cv)]
            draw_solution(ax, chargers, ue_score, n_open, pct)

            is_current = highlight_fn(rv, cv)
            border_col = "gold"  if is_current else "grey"
            border_lw  = 3.0    if is_current else 0.5
            for spine in ax.spines.values():
                spine.set_edgecolor(border_col)
                spine.set_linewidth(border_lw)

            ax.set_title(
                f"{row_label}={rv}  {col_label}={cv}\n"
                f"UE={ue_score:.2f}  ({pct:.1f}%)  "
                f"s={n_open}  c={n_char}",
                fontsize=7.5, pad=3,
            )

    # Shared colorbar on the right
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.12)
    cax = fig.add_axes([0.90, 0.15, 0.015, 0.65])
    cb  = fig.colorbar(sm, cax=cax)
    cb.set_label("Chargers / station", fontsize=8)
    cb.set_ticks(range(1, MAX_CHARGERS + 1))
    cb.ax.tick_params(labelsize=7)

    fig.savefig(OUT_DIR / fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Panel saved: {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# Grid 1: alpha_wait × noopt_cost  (penalty_dist = FIX_PENALTY_DIST)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"Grid 1 — alpha_wait × noopt   (penalty_dist={FIX_PENALTY_DIST})")
print(f"{'='*65}")

grid1 = {}
t0 = time.perf_counter()
for aw, noopt in product(ALPHA_WAIT_VALS, NOOPT_VALS):
    label = f"alpha_wait={aw:5.1f}  noopt={noopt:5.1f}  pd={FIX_PENALTY_DIST}"
    U_sol, score, n_open, n_char, chargers = solve_combo(
        alpha_wait=aw, noopt=noopt, penalty_dist=FIX_PENALTY_DIST, label=label,
    )
    pct = 100.0 * score / total_demand
    grid1[(aw, noopt)] = (chargers, score, n_open, n_char, pct)

    # Individual high-res map
    aw_str, noopt_str = f"{aw}".replace(".", "p"), f"{noopt}".replace(".", "p")
    save_individual(
        chargers, score, n_open, n_char, pct,
        fname=f"g1_aw{aw_str}_noopt{noopt_str}.png",
        title=(f"alpha_wait={aw}  noopt={noopt}  penalty_dist={FIX_PENALTY_DIST}\n"
               f"UE={score:.3f}  ({pct:.1f}% served)  "
               f"stations={n_open}  chargers={n_char}"),
    )

print(f"\n  Grid 1 done in {time.perf_counter()-t0:.1f}s")

make_panel(
    grid1, ALPHA_WAIT_VALS, NOOPT_VALS,
    row_label="alpha_wait", col_label="noopt",
    fixed_label=f"penalty_dist={FIX_PENALTY_DIST}",
    fname="panel_grid1_aw_x_noopt.png",
    highlight_fn=lambda aw, noopt: (
        aw == CURRENT["alpha_wait"] and noopt == CURRENT["noopt"]
    ),
)

# ══════════════════════════════════════════════════════════════════════════════
# Grid 2: alpha_wait × penalty_dist  (noopt = FIX_NOOPT)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print(f"Grid 2 — penalty_dist × alpha_wait   (noopt={FIX_NOOPT})")
print(f"{'='*65}")

grid2 = {}
t0 = time.perf_counter()
for pd_val, aw in product(PENALTY_DIST_VALS, ALPHA_WAIT_VALS):
    label = f"pd={pd_val:5.1f}  alpha_wait={aw:5.1f}  noopt={FIX_NOOPT}"
    U_sol, score, n_open, n_char, chargers = solve_combo(
        alpha_wait=aw, noopt=FIX_NOOPT, penalty_dist=pd_val, label=label,
    )
    pct = 100.0 * score / total_demand
    grid2[(pd_val, aw)] = (chargers, score, n_open, n_char, pct)

    aw_str, pd_str = f"{aw}".replace(".", "p"), f"{pd_val}".replace(".", "p")
    save_individual(
        chargers, score, n_open, n_char, pct,
        fname=f"g2_aw{aw_str}_pd{pd_str}.png",
        title=(f"penalty_dist={pd_val}  alpha_wait={aw}  noopt={FIX_NOOPT}\n"
               f"UE={score:.3f}  ({pct:.1f}% served)  "
               f"stations={n_open}  chargers={n_char}"),
    )

print(f"\n  Grid 2 done in {time.perf_counter()-t0:.1f}s")

make_panel(
    grid2, PENALTY_DIST_VALS, ALPHA_WAIT_VALS,
    row_label="penalty_dist", col_label="alpha_wait",
    fixed_label=f"noopt={FIX_NOOPT}",
    fname="panel_grid2_pd_x_aw.png",
    highlight_fn=lambda pd_val, aw: (
        aw == CURRENT["alpha_wait"] and pd_val == CURRENT["pd"]
    ),
)

# ── Summary CSV ───────────────────────────────────────────────────────────────
rows = []
for (aw, noopt), (ch, sc, no, nc, pct) in grid1.items():
    rows.append(dict(grid=1, alpha_wait=aw, noopt=noopt,
                     penalty_dist=FIX_PENALTY_DIST,
                     ue_score=round(sc, 4), pct_served=round(pct, 2),
                     n_open=no, n_chargers=nc))
for (pd_val, aw), (ch, sc, no, nc, pct) in grid2.items():
    rows.append(dict(grid=2, alpha_wait=aw, noopt=FIX_NOOPT,
                     penalty_dist=pd_val,
                     ue_score=round(sc, 4), pct_served=round(pct, 2),
                     n_open=no, n_chargers=nc))

df_res = pd.DataFrame(rows)
csv_out = OUT_DIR / "sweep_results.csv"
df_res.to_csv(csv_out, index=False)
print(f"\nSummary CSV: {csv_out}")
print(df_res.to_string(index=False))
print(f"\nAll plots -> {OUT_DIR}")
print("Done.")
