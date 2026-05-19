"""
gate_experiment.py  —  Gate comparison: No-gate / Fixed-gate / Adaptive-gate

Design
------
Three ALNS-UE runs on the same instance, same time budget, same seed:

  No gate      : every D&R candidate → LS → full UE evaluation
  Fixed gate   : cheap proxy filter → skip LS+UE for bad candidates
  Adaptive gate: epsilon-greedy gate — after each rejection, flip with prob p
                 and run UE anyway (for label collection). p decays each
                 iteration from p_explore_init → p_min, so early exploration
                 is free and the gate becomes aggressive over time.

Outputs
-------
  results/gate/adaptive_trajectory.png   (score vs time, 3 runs)
  results/gate/adaptive_convergence.png  (skip rate + p_explore over iterations)
  results/gate/adaptive_summary.csv      (stats table)

Usage
-----
    python scripts/gate_experiment.py
    python scripts/gate_experiment.py --csv center_79_Monza_k400.csv --time 90
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
p.add_argument("--csv",           default="center_79_Monza_k400.csv")
p.add_argument("--T",             type=int,   default=6)
p.add_argument("--D",             type=float, default=2.0)
p.add_argument("--seed",          type=int,   default=11)
p.add_argument("--time",          type=float, default=90.0,
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
# Adaptive gate hyper-parameters
p.add_argument("--p-explore",  type=float, default=0.30,
               help="Starting exploration rate (flip prob after gate rejection)")
p.add_argument("--p-min",      type=float, default=0.02,
               help="Floor for exploration rate — never stop exploring entirely")
p.add_argument("--p-decay",    type=float, default=0.995,
               help="Multiplicative decay applied to p each iteration")
args = p.parse_args()

OUT_DIR = PROJECT_ROOT / "results" / "gate"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHARGERS  = 6
Q_CAP         = 20.0
CUMULATIVE    = True
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]
LS_MODES      = ("site_swap", "local_remove")

# ── Instance ─────────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "input"
df_raw    = pd.read_csv(DATA_DIR / args.csv)
coords_km = df_raw[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float) * 111.0
pop       = np.maximum(df_raw["Aggregated_Population"].to_numpy(float), 0.0)
M         = coords_km.shape[0]
P_T       = [max(4, M // 30)] * args.T

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
    Ij_int[j].append(i)
    Ji_int[i].append(j)
J_i_list = [sorted(Ji_int[i], key=lambda j: distIJ[i, j]) for i in range(M)]

inst = {
    "coords_I": coords_km, "coords_J": coords_km.copy(),
    "demand_IT": demand_TM, "distIJ": distIJ,
    "in_range": in_range, "Ji": Ji, "Ij": Ij, "M": M, "N": M,
}

print(f"Instance : {args.csv}")
print(f"N={M}  T={args.T}  D={args.D}km  budget/period={P_T[0]}  time={args.time}s")

# ── UEEvaluator (full eval) ───────────────────────────────────────────────────
ue_mu   = args.ue_cap_per_server + 1e-4
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


# ── Fixed-gate ALNS loop ──────────────────────────────────────────────────────

def run_alns(label: str, gate_frac: float, time_budget: float, seed: int):
    """
    gate_frac=0.0 → gate disabled (evaluate everything)
    gate_frac>0   → skip LS+UE if proxy < gate_frac * initial_proxy
    """
    rng = np.random.default_rng(seed + 100)

    U_curr      = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=seed)
    score_curr  = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    proxy_init = proxy_fn(U_curr)
    proxy_gate = gate_frac * proxy_init if gate_frac > 0 else 0.0

    t_vec, score_vec = [0.0], [score_best]
    full_eval_count  = 1
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

        if proxy_gate > 0:
            p_val = proxy_fn(U_recon)
            if p_val < proxy_gate:
                skip_count += 1
                continue

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
        "forced_eval_count": 0,
        "new_best_count":  new_best_count,
        "best_score":      score_best,
        "greedy_score":    score_curr,
    }


# ── Adaptive-gate ALNS loop ───────────────────────────────────────────────────

def run_alns_adaptive(
    label: str,
    gate_frac: float,
    time_budget: float,
    seed: int,
    p_explore_init: float = 0.30,
    p_min: float = 0.02,
    decay: float = 0.995,
):
    """
    Epsilon-greedy gate with decaying exploration:

      Step 1 — gate_labels, p_explore, p_min, decay initialised here.
      Step 2 — on each gate rejection flip with prob p_explore:
                  heads → forced UE eval (label collected, no incumbent update)
                  tails → true skip (saves UE cost)
      Step 3 — decay p_explore = max(p_min, p_explore * decay) every iteration.

    With decay=0.995 and 1000 iters, p goes 0.30 → ~0.07 (still above p_min=0.02).
    Early on the gate is permissive; later it becomes aggressive.
    """
    rng = np.random.default_rng(seed + 100)

    U_curr     = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=seed)
    score_curr = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    proxy_init = proxy_fn(U_curr)
    proxy_gate = gate_frac * proxy_init if gate_frac > 0 else 0.0

    # ── Step 1: initialise gate state ────────────────────────────────────────
    gate_labels = []          # (proxy_score, ue_score) from forced evals
    p_explore   = p_explore_init

    t_vec, score_vec = [0.0], [score_best]
    p_history     = []        # p_explore at each iteration (for convergence plot)
    skip_per_iter = []        # 1 = true skip, 0 = passed gate or forced-eval'd

    full_eval_count   = 1
    skip_count        = 0
    forced_eval_count = 0
    iter_count        = 0
    new_best_count    = 0

    print(f"\n{'='*60}")
    print(f"  {label}   gate_frac={gate_frac}   proxy_gate={proxy_gate:.3f}")
    print(f"  p_explore_init={p_explore_init}  p_min={p_min}  decay={decay}")
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

        # ── Step 2: gate check with epsilon-greedy exploration ────────────────
        gate_rejected = False
        if proxy_gate > 0:
            p_val = proxy_fn(U_recon)
            if p_val < proxy_gate:
                gate_rejected = True
                if rng.random() < p_explore:
                    # Forced eval — collect label, do NOT update incumbent
                    ue_score = ue_fn(U_recon)
                    gate_labels.append((p_val, ue_score))
                    forced_eval_count += 1
                    full_eval_count   += 1
                    skip_per_iter.append(0)   # explored, not skipped
                else:
                    skip_count += 1
                    skip_per_iter.append(1)   # true skip

        if not gate_rejected:
            skip_per_iter.append(0)

        # ── Step 3: decay p at the bottom of every iteration ─────────────────
        p_explore = max(p_min, p_explore * decay)
        p_history.append(p_explore)

        if gate_rejected:
            continue   # rejected candidates never update incumbent

        # ── Normal path: LS + full eval ───────────────────────────────────────
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

        if iter_count % 20 == 0:
            print(f"  t={elapsed:6.1f}s  iter={iter_count:4d}  "
                  f"best={score_best:.2f}  skips={skip_count}  "
                  f"forced={forced_eval_count}  labels={len(gate_labels)}  "
                  f"p={p_explore:.4f}",
                  flush=True)

    total_time = time.perf_counter() - t0
    print(f"\n  Done   time={total_time:.1f}s  iters={iter_count}  "
          f"skips={skip_count}  forced={forced_eval_count}  "
          f"full_evals={full_eval_count}  new_bests={new_best_count}  "
          f"best={score_best:.3f}  gate_labels={len(gate_labels)}")

    return {
        "label":             label,
        "gate_frac":         gate_frac,
        "t_vec":             t_vec,
        "score_vec":         score_vec,
        "total_time":        total_time,
        "iter_count":        iter_count,
        "skip_count":        skip_count,
        "skip_rate":         skip_count / max(iter_count, 1),
        "full_eval_count":   full_eval_count,
        "forced_eval_count": forced_eval_count,
        "new_best_count":    new_best_count,
        "best_score":        score_best,
        "greedy_score":      score_curr,
        "gate_labels":       gate_labels,
        "p_history":         p_history,
        "skip_per_iter":     skip_per_iter,
    }


# ── Run three configurations ──────────────────────────────────────────────────
result_nogate = run_alns(
    "No gate",    gate_frac=0.0,            time_budget=args.time, seed=args.seed,
)
result_fixed = run_alns(
    "Fixed gate", gate_frac=args.gate_frac, time_budget=args.time, seed=args.seed,
)
result_adaptive = run_alns_adaptive(
    "Adaptive gate", gate_frac=args.gate_frac, time_budget=args.time, seed=args.seed,
    p_explore_init=args.p_explore, p_min=args.p_min, decay=args.p_decay,
)

rows = [result_nogate, result_fixed, result_adaptive]


# ── Summary table ─────────────────────────────────────────────────────────────
summary = pd.DataFrame([{
    "config":          r["label"],
    "gate_frac":       r["gate_frac"],
    "time_s":          round(r["total_time"], 1),
    "iterations":      r["iter_count"],
    "skips":           r["skip_count"],
    "skip_rate_%":     round(100 * r["skip_rate"], 1),
    "full_evals":      r["full_eval_count"],
    "gate_labels":     r["forced_eval_count"],
    "new_bests":       r["new_best_count"],
    "best_score":      round(r["best_score"], 3),
} for r in rows])

print("\n" + "=" * 70)
print(summary.to_string(index=False))
print("=" * 70)

csv_path = OUT_DIR / "adaptive_summary.csv"
summary.to_csv(csv_path, index=False)
print(f"\nSummary -> {csv_path}")


# ── Shared style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9, "grid.alpha": 0.20,
    "grid.linestyle": "--", "grid.linewidth": 0.6,
})

COLORS = {
    "No gate":       "#2471A3",
    "Fixed gate":    "#C0392B",
    "Adaptive gate": "#27AE60",
}
LSTYLE = {
    "No gate":       "-",
    "Fixed gate":    "--",
    "Adaptive gate": "-.",
}


# ── Plot 1: Trajectory — score vs time (3 runs) ───────────────────────────────
fig, (ax_traj, ax_tbl) = plt.subplots(
    1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 1]}
)
fig.suptitle(
    f"{Path(args.csv).stem}   $N={M}$, $T={args.T}$,"
    f"  time budget = {args.time:.0f} s",
    fontsize=11, y=1.02,
)

ax_traj.grid(axis="y")
for r in rows:
    c  = COLORS[r["label"]]
    ls = LSTYLE[r["label"]]
    ax_traj.plot(
        r["t_vec"], r["score_vec"],
        color=c, linewidth=2.0, linestyle=ls,
        label=f"{r['label']}  (iters={r['iter_count']}, best={r['best_score']:.1f})",
    )
    jump_t = [r["t_vec"][k] for k in range(1, len(r["t_vec"]))
              if r["score_vec"][k] > r["score_vec"][k - 1]]
    jump_s = [r["score_vec"][k] for k in range(1, len(r["t_vec"]))
              if r["score_vec"][k] > r["score_vec"][k - 1]]
    ax_traj.scatter(jump_t, jump_s, color=c, s=30, zorder=4,
                    edgecolors="white", linewidths=0.8)

ax_traj.axhline(
    result_nogate["score_vec"][0], color="#888", linewidth=1.2, linestyle=":",
    label=f"Greedy  {result_nogate['score_vec'][0]:.1f}",
)
ax_traj.set_xlabel("Wall-clock time (s)", fontsize=11)
ax_traj.set_ylabel("UE score (demand served)", fontsize=11)
ax_traj.set_xlim(0, args.time)
ax_traj.legend(fontsize=8.5, frameon=False, loc="lower right")
ax_traj.set_title("Incumbent trajectory vs time", fontsize=10, pad=6)

# Summary table
ax_tbl.axis("off")
table_data = [
    ["",            "No gate",                       "Fixed gate",                      "Adaptive"],
    ["Iterations",  str(result_nogate["iter_count"]),  str(result_fixed["iter_count"]),   str(result_adaptive["iter_count"])],
    ["Skips",       "—",                               str(result_fixed["skip_count"]),   str(result_adaptive["skip_count"])],
    ["Skip rate",   "—",                               f"{100*result_fixed['skip_rate']:.0f}%",  f"{100*result_adaptive['skip_rate']:.0f}%"],
    ["Full evals",  str(result_nogate["full_eval_count"]), str(result_fixed["full_eval_count"]), str(result_adaptive["full_eval_count"])],
    ["Gate labels", "—",                               "—",                               str(result_adaptive["forced_eval_count"])],
    ["New bests",   str(result_nogate["new_best_count"]), str(result_fixed["new_best_count"]),   str(result_adaptive["new_best_count"])],
    ["Best score",  f"{result_nogate['best_score']:.2f}", f"{result_fixed['best_score']:.2f}",  f"{result_adaptive['best_score']:.2f}"],
]
tbl = ax_tbl.table(
    cellText=table_data[1:], colLabels=table_data[0],
    cellLoc="center", loc="center", bbox=[0.0, 0.05, 1.0, 0.90],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(fontweight="bold")
    elif col == 3 and row > 0:
        cell.set_facecolor("#f0faf0")   # adaptive column
    elif col == 2 and row > 0:
        cell.set_facecolor("#fdf3f3")   # fixed gate column
    else:
        cell.set_facecolor("white")
ax_tbl.set_title("Summary", fontsize=10, pad=6)

plt.tight_layout()
traj_path = OUT_DIR / "adaptive_trajectory.png"
fig.savefig(traj_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Trajectory plot  -> {traj_path}")


# ── Plot 2: Convergence — skip rate + p_explore decay ────────────────────────
skip_arr = np.array(result_adaptive["skip_per_iter"], dtype=float)
p_arr    = np.array(result_adaptive["p_history"],     dtype=float)
iters    = np.arange(1, len(skip_arr) + 1)

WIN = 30
if len(skip_arr) >= WIN:
    rolling_skip = np.convolve(skip_arr, np.ones(WIN) / WIN, mode="valid")
    roll_iters   = iters[WIN - 1:]
else:
    rolling_skip = np.cumsum(skip_arr) / iters
    roll_iters   = iters

fig2, (ax_skip, ax_p) = plt.subplots(1, 2, figsize=(13, 4.5))
fig2.suptitle(
    f"Adaptive gate convergence — {Path(args.csv).stem}  "
    f"p_init={args.p_explore}  decay={args.p_decay}  p_min={args.p_min}",
    fontsize=11, y=1.02,
)

ax_skip.plot(roll_iters, rolling_skip * 100, color="#27AE60", linewidth=2.0)
ax_skip.set_xlabel("Iteration", fontsize=11)
ax_skip.set_ylabel("Skip rate — 30-iter rolling avg (%)", fontsize=11)
ax_skip.set_title("Skip rate increases as p_explore decays", fontsize=10, pad=6)
ax_skip.grid(axis="y", alpha=0.2, linestyle="--")
ax_skip.set_ylim(0, 105)
ax_skip.spines["top"].set_visible(False)
ax_skip.spines["right"].set_visible(False)

ax_p.plot(iters, p_arr * 100, color="#8E44AD", linewidth=2.0)
ax_p.axhline(args.p_min * 100, color="#555", linewidth=1.0, linestyle=":",
             label=f"p_min = {args.p_min}")
ax_p.set_xlabel("Iteration", fontsize=11)
ax_p.set_ylabel("Exploration rate p_explore (%)", fontsize=11)
ax_p.set_title("p_explore decay over iterations", fontsize=10, pad=6)
ax_p.grid(axis="y", alpha=0.2, linestyle="--")
ax_p.legend(fontsize=9, frameon=False)
ax_p.spines["top"].set_visible(False)
ax_p.spines["right"].set_visible(False)

plt.tight_layout()
conv_path = OUT_DIR / "adaptive_convergence.png"
fig2.savefig(conv_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig2)
plt.rcdefaults()
print(f"Convergence plot -> {conv_path}")

ue_eval.dispose()
print("Done.")
