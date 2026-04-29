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
p.add_argument("--ue-noopt",            type=float, default=30.0)
p.add_argument("--ue-max-range",        type=float, default=2.0)
p.add_argument("--ue-penalty-distance", type=float, default=5.0)
p.add_argument("--ue-cap-per-server",   type=float, default=4.0)
p.add_argument("--tag", default="",
               help="Optional suffix appended to output directory name")
args = p.parse_args()

DATA_DIR = PROJECT_ROOT / "data" / "input"

instance_stem = Path(args.csv).stem
dir_name = instance_stem + (f"_{args.tag}" if args.tag else "")
OUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "alns_ue" / dir_name
OUT_DIR.mkdir(parents=True, exist_ok=True)

P_T           = [8] * args.T
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

# ── Objective trajectory plot (main + zoomed inset) ──────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
fig.suptitle(
    f"{args.csv}  N={M}  T={args.T}  D={args.D}km  |  "
    f"iters={len(traj_iter)-1}  frac={args.frac_remove}  ls_moves={args.ls_moves}  "
    f"eps={args.accept_eps}  elapsed={elapsed:.0f}s  |  "
    f"UE: alpha_w={args.ue_alpha_wait}  range={args.ue_max_range}km  "
    f"pd={args.ue_penalty_distance}  cap/s={args.ue_cap_per_server}",
    fontsize=8.5,
)

# ── Main panel: full exploration trajectory ───────────────────────────────────
ax.plot(traj_iter, traj_score_curr, color="steelblue", linewidth=0.8, alpha=0.45,
        linestyle="-", label="Current UE (U_curr)", zorder=2)
ax.plot(traj_iter, traj_score_best, color="crimson", linewidth=2.5,
        label="Best UE (U_best)", zorder=3)

for inc_it, inc_score, _ in incumbents:
    ax.axvline(inc_it, color="crimson", linewidth=0.8, linestyle=":", alpha=0.4)
    ax.scatter([inc_it], [inc_score], s=50, color="crimson", zorder=5)

ax.set_xlabel("ALNS iteration", fontsize=9)
ax.set_ylabel("UE score", fontsize=9)
ax.set_title(
    "Objective trajectory — blue = current solution (exploration), "
    "red = best-ever  |  inset: best-score staircase zoomed",
    fontsize=9,
)
ax.legend(fontsize=8, loc="lower right")
ax.set_xlim(0, traj_iter[-1])

# ── Zoomed inset: best-score staircase, right side ───────────────────────────
axins = ax.inset_axes([0.57, 0.06, 0.41, 0.88])

# Only the staircase — no noisy current-solution line
axins.plot(traj_iter, traj_score_best, color="crimson", linewidth=2.2, zorder=3)

best_scores = [inc[1] for inc in incumbents]
y_margin = 0.12
y_lo = min(best_scores) - y_margin
y_hi = max(best_scores) + y_margin
axins.set_ylim(y_lo, y_hi)
axins.set_xlim(0, traj_iter[-1])

# Vertical drop lines + numbered dots at each step
for k, (inc_it, inc_score, inc_mode) in enumerate(incumbents):
    axins.axvline(inc_it, color="crimson", linewidth=0.8, linestyle=":", alpha=0.4)

axins.scatter(
    [inc[0] for inc in incumbents],
    [inc[1] for inc in incumbents],
    s=55, color="crimson", zorder=5,
)

# Alternate #labels above/below each dot so they never overlap
for k, (inc_it, inc_score, _) in enumerate(incumbents):
    va  = "bottom" if k % 2 == 0 else "top"
    dy  =  0.025   if k % 2 == 0 else -0.025
    axins.text(inc_it, inc_score + dy, f"#{k}",
               fontsize=7.5, color="crimson", ha="center", va=va, fontweight="bold")

# Monospace detail table in the lower-left corner of the inset
rows = ["#   iter   score    delta       mode"]
rows.append("-" * 38)
for k, (inc_it, inc_score, inc_mode) in enumerate(incumbents):
    delta_str = f"+{inc_score - incumbents[k-1][1]:.3f}" if k > 0 else "baseline"
    rows.append(f"{k}  {inc_it:>5d}  {inc_score:.3f}  {delta_str:<10}  {inc_mode}")

axins.text(
    0.02, 0.02, "\n".join(rows),
    transform=axins.transAxes,
    fontsize=5.8, color="crimson", family="monospace",
    va="bottom", ha="left",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
              alpha=0.88, edgecolor="crimson", lw=0.6),
)

axins.set_xlabel("iteration", fontsize=7)
axins.set_ylabel("best UE", fontsize=7)
axins.tick_params(labelsize=6.5)
axins.set_title("Best-score staircase (zoomed)", fontsize=7.5, pad=3)
axins.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter("%.3f"))

# Shaded band on the main panel showing the inset's y-range
ax.axhspan(y_lo, y_hi, alpha=0.07, color="crimson", zorder=0)
ax.annotate("inset region", xy=(traj_iter[-1] * 0.28, (y_lo + y_hi) / 2),
            fontsize=7.5, color="crimson", alpha=0.6, va="center")

fig.tight_layout()
traj_fname = "alns_trajectory.png"
fig.savefig(OUT_DIR / traj_fname, dpi=150, bbox_inches="tight")
plt.close(fig)
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
