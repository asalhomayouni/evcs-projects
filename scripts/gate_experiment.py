"""
gate_experiment.py  —  Preliminary gate comparison on Palermo N=1025

Design
------
Two ALNS-UE runs on the same instance, same time budget, same seed:

  No gate  : every D&R candidate → LS → full UE evaluation
  With gate: cheap proxy filter first → skip LS+UE for bad candidates

Hypothesis
----------
Gate → fewer expensive UE calls per unit time
     → more accepted iterations in the same wall-clock budget
     → similar or better incumbent quality

Outputs
-------
  results/gate/gate_vs_nogate_trajectory.png   (score vs time)
  results/gate/gate_vs_nogate_summary.csv      (stats table)

Usage
-----
    python scripts/gate_experiment.py
    python scripts/gate_experiment.py --csv center_79_Monza_k400.csv --time 60
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

from evcs.geom          import build_arcs
from evcs.destroy       import destroy_multi_u
from evcs.greedy        import reconstruct_u_dict_fast
from evcs.methods       import greedy_schedule_multi_from_variants
from evcs.local_search  import local_search_u_proxy
from evcs.proxy         import evaluate_u_numpy_greedy_binary
from evcs.ue_evaluator  import UEEvaluator

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--csv",           default="center_710_Palermo_k1025.csv")
p.add_argument("--T",             type=int,   default=6)
p.add_argument("--D",             type=float, default=2.0)
p.add_argument("--seed",          type=int,   default=11)
p.add_argument("--time",          type=float, default=120.0,
               help="Wall-clock time budget per run (s)")
p.add_argument("--gate-frac",     type=float, default=0.90,
               help="proxy_gate = gate_frac × initial_proxy; 0=disabled")
p.add_argument("--ls-moves",      type=int,   default=8)
p.add_argument("--frac-remove",   type=float, default=0.20)
p.add_argument("--ue-alpha-wait",       type=float, default=10.0)
p.add_argument("--ue-noopt",            type=float, default=10.0)
p.add_argument("--ue-penalty-distance", type=float, default=5.0)
p.add_argument("--ue-cap-per-server",   type=float, default=4.0)
p.add_argument("--ue-max-range",        type=float, default=2.0)
args = p.parse_args()

OUT_DIR = PROJECT_ROOT / "results" / "gate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHARGERS  = 6
Q_CAP         = 20.0
CUMULATIVE    = True
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]
LS_MODES      = ("site_swap", "local_remove")

# ── Instance ─────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "input"
df_raw     = pd.read_csv(DATA_DIR / args.csv)
coords_km  = df_raw[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float) * 111.0
pop        = np.maximum(df_raw["Aggregated_Population"].to_numpy(float), 0.0)
M          = coords_km.shape[0]
P_T        = [max(4, M // 30)] * args.T   # same budget rule as plot_alns_ue

rng_inst  = np.random.default_rng(args.seed)
base      = (pop / pop.sum()) * M
demand_TM = np.zeros((args.T, M))
for t in range(args.T):
    sf = 1.0 + 0.2 * np.sin(2 * np.pi * t / args.T)
    demand_TM[t] = np.maximum(0.0, base * sf * (1.0 + rng_inst.normal(0, 0.05, M)))

distIJ, in_range, Ji, Ij = build_arcs(coords_km, coords_km, D=args.D, forbid_self=False)
Ij_int = {j: [] for j in range(M)}
Ji_int = {i: [] for i in range(M)}
for (i, j) in in_range:
    Ij_int[j].append(i); Ji_int[i].append(j)
J_i_list = [sorted(Ji_int[i], key=lambda j: distIJ[i, j]) for i in range(M)]

inst = {
    "coords_I": coords_km, "coords_J": coords_km.copy(),
    "demand_IT": demand_TM, "distIJ": distIJ,
    "in_range": in_range, "Ji": Ji, "Ij": Ij, "M": M, "N": M,
}

print(f"Instance : {args.csv}")
print(f"N={M}  T={args.T}  D={args.D}km  budget/period={P_T[0]}  time={args.time}s")

# ── UEEvaluator (full eval) ───────────────────────────────────────────────────
ue_mu = args.ue_cap_per_server + 1e-4
ue_eval = UEEvaluator(
    N=M, d=demand_TM[0], tau=distIJ, mu=ue_mu, s_max=MAX_CHARGERS,
    noopt_cost=args.ue_noopt, alpha_wait=args.ue_alpha_wait, N_bp=50,
    max_range=args.ue_max_range, penalty_distance=args.ue_penalty_distance,
)

def ue_fn(U):
    s, _ = ue_eval.evaluate(U, T=args.T, cumulative_install=CUMULATIVE)
    return float(s)

def proxy_fn(U):
    return float(evaluate_u_numpy_greedy_binary(
        U, demand_TM, J_i_list, distIJ, Q_CAP, args.T, M, CUMULATIVE
    ))


# ── Core ALNS loop with optional proxy gate ───────────────────────────────────

def run_alns(label: str, gate_frac: float, time_budget: float, seed: int):
    """
    Returns a dict with trajectory lists and summary stats.
    gate_frac=0.0 → gate disabled (evaluate everything)
    gate_frac>0   → skip LS+UE if proxy < gate_frac * initial_proxy
    """
    rng = np.random.default_rng(seed + 100)

    U_curr = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=seed)
    score_curr = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    # Initial proxy for gate threshold
    proxy_init  = proxy_fn(U_curr)
    proxy_gate  = gate_frac * proxy_init if gate_frac > 0 else 0.0

    # Trajectory
    t_vec, score_vec = [0.0], [score_best]
    full_eval_count  = 1          # counted greedy init above
    skip_count       = 0
    iter_count       = 0
    new_best_count   = 0

    print(f"\n{'='*60}")
    print(f"  {label}   gate_frac={gate_frac}   proxy_gate={proxy_gate:.3f}")
    print(f"{'='*60}")

    t0 = time.perf_counter()

    while True:
        elapsed = time.perf_counter() - t0
        if elapsed >= time_budget:
            break

        iter_count += 1
        base = U_curr if rng.random() < 0.7 else U_best
        mode = str(rng.choice(DESTROY_MODES))

        U_partial, _ = destroy_multi_u(
            base, inst, rng, P_T, args.frac_remove, mode,
            site_cap=MAX_CHARGERS, cumulative_install=CUMULATIVE,
        )
        U_recon, _ = reconstruct_u_dict_fast(
            U_partial, demand_TM, P_T, Ij_int,
            U_cap=MAX_CHARGERS, Q=Q_CAP, rng=rng, cumulative_install=CUMULATIVE,
        )

        # ── Gate: cheap proxy check ───────────────────────────────────────────
        if proxy_gate > 0:
            p_val = proxy_fn(U_recon)
            if p_val < proxy_gate:
                skip_count += 1
                continue            # skip LS + UE entirely

        # ── Local search (UE as proxy_fn — every LS move is a full UE call) ──
        visited, ue_scores, _ = local_search_u_proxy(
            U_recon, inst, rng, P_T,
            demand_TM, J_i_list, distIJ,
            Q_CAP, args.T, M, CUMULATIVE, MAX_CHARGERS,
            args.ls_moves, 0.08, LS_MODES,
            Ij_int=Ij_int, collect_visited=True, proxy_fn=ue_fn,
        )
        full_eval_count += len(visited)

        best_idx   = int(np.argmax(ue_scores))
        iter_score = float(ue_scores[best_idx])
        iter_U     = visited[best_idx]

        if iter_score >= score_curr:
            U_curr, score_curr = dict(iter_U), iter_score

        if iter_score > score_best:
            U_best, score_best = dict(iter_U), iter_score
            new_best_count += 1

        elapsed = time.perf_counter() - t0
        t_vec.append(elapsed)
        score_vec.append(score_best)

        if iter_count % 20 == 0 or iter_score > score_best:
            print(f"  t={elapsed:6.1f}s  iter={iter_count:4d}  "
                  f"best={score_best:.2f}  skips={skip_count}  "
                  f"full_evals={full_eval_count}",
                  flush=True)

    total_time = time.perf_counter() - t0
    print(f"\n  Done   time={total_time:.1f}s  iters={iter_count}  "
          f"skips={skip_count}  full_evals={full_eval_count}  "
          f"new_bests={new_best_count}  best={score_best:.3f}")

    return {
        "label":           label,
        "gate_frac":       gate_frac,
        "t_vec":           t_vec,
        "score_vec":       score_vec,
        "total_time":      total_time,
        "iter_count":      iter_count,
        "skip_count":      skip_count,
        "skip_rate":       skip_count / max(iter_count, 1),
        "full_eval_count": full_eval_count,
        "new_best_count":  new_best_count,
        "best_score":      score_best,
        "greedy_score":    score_curr,
    }


# ── Run both configurations ───────────────────────────────────────────────────
result_nogate = run_alns("No gate",   gate_frac=0.0,          time_budget=args.time, seed=args.seed)
result_gate   = run_alns("With gate", gate_frac=args.gate_frac, time_budget=args.time, seed=args.seed)


# ── Summary table ─────────────────────────────────────────────────────────────
rows = [result_nogate, result_gate]
summary = pd.DataFrame([{
    "config":          r["label"],
    "gate_frac":       r["gate_frac"],
    "time_s":          round(r["total_time"], 1),
    "iterations":      r["iter_count"],
    "skips":           r["skip_count"],
    "skip_rate_%":     round(100 * r["skip_rate"], 1),
    "full_evals":      r["full_eval_count"],
    "new_bests":       r["new_best_count"],
    "best_score":      round(r["best_score"], 3),
} for r in rows])

print("\n" + "="*70)
print(summary.to_string(index=False))
print("="*70)

csv_path = OUT_DIR / "gate_vs_nogate_summary.csv"
summary.to_csv(csv_path, index=False)
print(f"\nSummary -> {csv_path}")


# ── Trajectory plot ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "grid.alpha": 0.20,
    "grid.linestyle": "--", "grid.linewidth": 0.6,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                         gridspec_kw={"width_ratios": [2, 1]})
fig.suptitle(
    f"{Path(args.csv).stem}   $N={M}$,  $T={args.T}$,  "
    f"time budget = {args.time:.0f} s   [preliminary]",
    fontsize=11, y=1.02,
)

# ── Left: score vs time ───────────────────────────────────────────────────────
ax = axes[0]
ax.grid(axis="y")

COLORS = {"No gate": "#2471A3", "With gate": "#C0392B"}

for r in rows:
    c = COLORS[r["label"]]
    ax.plot(r["t_vec"], r["score_vec"],
            color=c, linewidth=2.0,
            label=f"{r['label']}  (iters={r['iter_count']}, best={r['best_score']:.1f})")
    # Mark new-best events (jumps in score_vec)
    jump_t = [r["t_vec"][k] for k in range(1, len(r["t_vec"]))
               if r["score_vec"][k] > r["score_vec"][k-1]]
    jump_s = [r["score_vec"][k] for k in range(1, len(r["t_vec"]))
               if r["score_vec"][k] > r["score_vec"][k-1]]
    ax.scatter(jump_t, jump_s, color=c, s=30, zorder=4, edgecolors="white", linewidths=0.8)

# Greedy baseline
ax.axhline(result_nogate["score_vec"][0],
           color="#888", linewidth=1.2, linestyle=":",
           label=f"Greedy  {result_nogate['score_vec'][0]:.1f}")

ax.set_xlabel("Wall-clock time (s)", fontsize=11)
ax.set_ylabel("UE score (demand served)", fontsize=11)
ax.set_xlim(0, args.time)
ax.legend(fontsize=8.5, frameon=False, loc="lower right")
ax.set_title("Incumbent trajectory vs time", fontsize=10, pad=6)

# ── Right: bar chart summary ──────────────────────────────────────────────────
ax2 = axes[1]
ax2.axis("off")

table_data = [
    ["", "No gate", "With gate"],
    ["Iterations",  str(result_nogate["iter_count"]),  str(result_gate["iter_count"])],
    ["Skips",       "—",                               str(result_gate["skip_count"])],
    ["Skip rate",   "—",                               f"{100*result_gate['skip_rate']:.0f}%"],
    ["Full evals",  str(result_nogate["full_eval_count"]), str(result_gate["full_eval_count"])],
    ["New bests",   str(result_nogate["new_best_count"]),  str(result_gate["new_best_count"])],
    ["Best score",  f"{result_nogate['best_score']:.2f}", f"{result_gate['best_score']:.2f}"],
]

tbl = ax2.table(
    cellText=table_data[1:], colLabels=table_data[0],
    cellLoc="center", loc="center",
    bbox=[0.0, 0.1, 1.0, 0.8],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(fontweight="bold")
    elif col == 2 and row > 0:
        cell.set_facecolor("#fdf3f3")   # highlight gate column
    else:
        cell.set_facecolor("white")

ax2.set_title("Summary  (preliminary)", fontsize=10, pad=6)

plt.tight_layout()
plot_path = OUT_DIR / "gate_vs_nogate_trajectory.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
plt.rcdefaults()
print(f"Plot     -> {plot_path}")

ue_eval.dispose()
print("Done.")
