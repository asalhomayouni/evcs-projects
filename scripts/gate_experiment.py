
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
from scipy.stats import norm as sp_norm

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
p.add_argument("--alpha",      type=float, default=0.10,
               help="Significance level for statistical gate: "
                    "P(candidate beats best) >= alpha. alpha=0.5 recovers rolling-mean.")
p.add_argument("--floor-frac", type=float, default=None,
               help="Floor multiplier for adaptive gate threshold "
                    "(default: same as --gate-frac). Set higher than gate-frac "
                    "to give calibration room above the fixed gate floor.")
p.add_argument("--out-tag",   type=str,   default=None,
               help="Sub-folder tag for outputs (default: timestamp). "
                    "Prevents overwriting previous results.")
args = p.parse_args()

import datetime as _dt
_tag = args.out_tag if args.out_tag else _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / "gate" / _tag
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
    threshold_method: str = "rolling_mean",
    alpha: float = 0.10,
    floor_frac: float = None,
):
    """
    Epsilon-greedy gate with three threshold strategies:

      rolling_mean : proxy_gate = proxy_best / rolling_mean(LS improvement ratio)
      statistical  : proxy_gate = proxy_best / norm.ppf(1-alpha, mu_ratio, sigma_ratio)
                     (shown to be equivalent to rolling_mean when sigma is tiny)
      calibration  : fit UE = b0 + b1*proxy on ALL observed (proxy, UE) pairs;
                     pass candidate p if P(b0 + b1*p + eps > UE_best) >= alpha,
                     i.e. proxy_gate = (UE_best + sigma_reg * norm.ppf(alpha) - b0) / b1.
                     This is the principled threshold: it uses the actual proxy→UE
                     relationship, not the LS improvement ratio.
    """
    N_WARMUP        = 10    # LS pairs needed before rolling/stat threshold activates
    EMA_WINDOW      = 30    # rolling window for improvement-ratio methods
    N_WARMUP_CALIB  = 20    # (proxy, UE) pairs needed before calibration activates
    CALIB_WINDOW    = 100   # pairs used for regression (most recent)

    rng = np.random.default_rng(seed + 100)

    U_curr     = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=seed)
    score_curr = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    proxy_init   = proxy_fn(U_curr)
    proxy_gate   = gate_frac * proxy_init if gate_frac > 0 else 0.0  # fixed during warmup
    _floor_frac  = floor_frac if floor_frac is not None else gate_frac
    proxy_floor  = _floor_frac * proxy_init  # adaptive threshold never goes below this

    gate_labels         = []   # (proxy, ue) from forced evals of rejected candidates
    all_proxy_ue_log    = []   # (proxy, ue) from ALL full evaluations (for calibration)
    p_explore           = p_explore_init

    ls_log              = []
    avg_improvement     = 1.0
    threshold_history   = []   # (iter, proxy_gate, threshold_value)

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

        # ── Gate check with epsilon-greedy exploration ───────────────────────
        # Always evaluate proxy when gate is active — we need it for calibration
        # data collection even when the candidate passes the gate.
        p_val = proxy_fn(U_recon) if proxy_gate > 0 else None

        gate_rejected = False
        if proxy_gate > 0 and p_val < proxy_gate:
            gate_rejected = True
            if rng.random() < p_explore:
                ue_score = ue_fn(U_recon)
                gate_labels.append((p_val, ue_score))
                all_proxy_ue_log.append((p_val, ue_score))
                forced_eval_count += 1
                full_eval_count   += 1
                skip_per_iter.append(0)
            else:
                skip_count += 1
                skip_per_iter.append(1)

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

        # Record LS improvement pair (for rolling/statistical threshold)
        ue_before = float(ue_scores[0])
        ue_after  = float(iter_score)
        if ue_before > 1e-9:
            ls_log.append((ue_before, ue_after))

        # Record (proxy, UE) pair for calibration — proxy of U_recon before LS,
        # UE of the best solution LS found from that starting point.
        if p_val is not None:
            all_proxy_ue_log.append((p_val, iter_score))

        # ── Update proxy_gate once we have enough data ────────────────────────
        enough_ls   = len(ls_log) >= N_WARMUP
        enough_calib = len(all_proxy_ue_log) >= N_WARMUP_CALIB
        can_update  = gate_frac > 0 and (
            (threshold_method == "calibration" and enough_calib) or
            (threshold_method != "calibration" and enough_ls)
        )

        if can_update:
            if threshold_method == "calibration":
                window_c = all_proxy_ue_log[-CALIB_WINDOW:]
                prx = np.array([x[0] for x in window_c])
                ues = np.array([x[1] for x in window_c])

                use_fallback = True
                if np.std(prx) > 1e-9 and len(prx) >= 4:
                    beta1, beta0 = np.polyfit(prx, ues, 1)
                    if beta1 > 1e-9:
                        residuals = ues - (beta0 + beta1 * prx)
                        sigma_reg = max(float(np.std(residuals, ddof=2)), 1e-9)
                        # Pass candidate p if P(UE > score_best) >= alpha
                        # UE | p ~ N(beta0 + beta1*p, sigma_reg^2)
                        # => beta0 + beta1*p >= score_best + sigma_reg * norm.ppf(alpha)
                        # => p >= (score_best + sigma_reg * norm.ppf(alpha) - beta0) / beta1
                        z_alpha = float(sp_norm.ppf(alpha))
                        proxy_gate_adapt = (score_best + sigma_reg * z_alpha - beta0) / beta1
                        avg_improvement  = proxy_gate_adapt  # store for logging
                        use_fallback = False

                if use_fallback:
                    window_ls = ls_log[-EMA_WINDOW:]
                    ratios_w  = [a / b for b, a in window_ls if b > 1e-9]
                    avg_improvement  = float(np.mean(ratios_w)) if ratios_w else 1.0
                    proxy_best_now   = proxy_fn(U_best)
                    proxy_gate_adapt = proxy_best_now / avg_improvement

            else:
                window_ls = ls_log[-EMA_WINDOW:]
                ratios_w  = [a / b for b, a in window_ls if b > 1e-9]
                proxy_best_now = proxy_fn(U_best)

                if threshold_method == "statistical":
                    mu_r    = float(np.mean(ratios_w))
                    sigma_r = max(float(np.std(ratios_w, ddof=1)) if len(ratios_w) > 1 else 1e-9, 1e-9)
                    q_alpha = max(float(sp_norm.ppf(1.0 - alpha, loc=mu_r, scale=sigma_r)), 1e-6)
                    avg_improvement = q_alpha
                else:
                    avg_improvement = float(np.mean(ratios_w))

                proxy_gate_adapt = proxy_best_now / avg_improvement

            # Floor: never stricter than proxy_floor (= floor_frac * proxy_init).
            # Default floor_frac == gate_frac, but can be set higher to give
            # calibration room to operate above the fixed-gate baseline.
            proxy_gate = min(proxy_gate_adapt, proxy_floor)
            threshold_history.append((iter_count, proxy_gate, avg_improvement))

        if iter_score >= score_curr:
            U_curr, score_curr = dict(iter_U), iter_score

        if iter_score > score_best:
            U_best, score_best = dict(iter_U), iter_score
            new_best_count += 1

        elapsed = time.perf_counter() - t0
        t_vec.append(elapsed)
        score_vec.append(score_best)

        if iter_count % 20 == 0:
            if threshold_method == "calibration":
                thr_tag = f"calib_gate={avg_improvement:.3f}  calib_pairs={len(all_proxy_ue_log)}"
            elif threshold_method == "statistical":
                thr_tag = f"q_alpha={avg_improvement:.4f}"
            else:
                thr_tag = f"avg_ratio={avg_improvement:.4f}"
            print(f"  t={elapsed:6.1f}s  iter={iter_count:4d}  "
                  f"best={score_best:.2f}  skips={skip_count}  "
                  f"forced={forced_eval_count}  p={p_explore:.4f}  "
                  f"{thr_tag}  gate={proxy_gate:.3f}",
                  flush=True)

    total_time = time.perf_counter() - t0
    final_ratios  = [a / b for b, a in ls_log if b > 1e-9]
    mu_r_final    = float(np.mean(final_ratios))         if final_ratios           else float("nan")
    sigma_r_final = float(np.std(final_ratios, ddof=1))  if len(final_ratios) > 1  else float("nan")

    # Calibration: fit final regression on all collected (proxy, UE) pairs
    beta0_final = beta1_final = sigma_reg_final = float("nan")
    if threshold_method == "calibration" and len(all_proxy_ue_log) >= 4:
        prx_f = np.array([x[0] for x in all_proxy_ue_log])
        ues_f = np.array([x[1] for x in all_proxy_ue_log])
        if np.std(prx_f) > 1e-9:
            b1f, b0f = np.polyfit(prx_f, ues_f, 1)
            beta0_final, beta1_final = float(b0f), float(b1f)
            sigma_reg_final = float(np.std(ues_f - (b0f + b1f * prx_f), ddof=2))

    print(f"\n  Done   time={total_time:.1f}s  iters={iter_count}  "
          f"skips={skip_count}  forced={forced_eval_count}  "
          f"full_evals={full_eval_count}  new_bests={new_best_count}  "
          f"best={score_best:.3f}  gate_labels={len(gate_labels)}  "
          f"calib_pairs={len(all_proxy_ue_log)}  ls_pairs={len(ls_log)}")
    if final_ratios:
        print(f"  LS improvement ratio  mean={mu_r_final:.4f}  "
              f"sigma={sigma_r_final:.4f}")
    if threshold_method == "calibration" and not np.isnan(beta1_final):
        z_a = float(sp_norm.ppf(alpha))
        thr_f = (score_best + sigma_reg_final * z_a - beta0_final) / beta1_final
        print(f"  Calibration fit  beta0={beta0_final:.4f}  beta1={beta1_final:.6f}  "
              f"sigma_reg={sigma_reg_final:.4f}")
        print(f"  Final threshold = ({score_best:.3f} + {sigma_reg_final:.4f}*{z_a:.3f} "
              f"- {beta0_final:.4f}) / {beta1_final:.6f} = {thr_f:.3f}")

    return {
        "label":               label,
        "gate_frac":           gate_frac,
        "t_vec":               t_vec,
        "score_vec":           score_vec,
        "total_time":          total_time,
        "iter_count":          iter_count,
        "skip_count":          skip_count,
        "skip_rate":           skip_count / max(iter_count, 1),
        "full_eval_count":     full_eval_count,
        "forced_eval_count":   forced_eval_count,
        "new_best_count":      new_best_count,
        "best_score":          score_best,
        "greedy_score":        score_curr,
        "gate_labels":         gate_labels,
        "p_history":           p_history,
        "skip_per_iter":       skip_per_iter,
        "ls_log":              ls_log,
        "threshold_history":   threshold_history,
        "avg_improvement":     avg_improvement,
        "threshold_method":    threshold_method,
        "alpha":               alpha,
        "mu_r_final":          mu_r_final,
        "sigma_r_final":       sigma_r_final,
        "all_proxy_ue_log":    all_proxy_ue_log,
        "beta0_final":         beta0_final,
        "beta1_final":         beta1_final,
        "sigma_reg_final":     sigma_reg_final,
    }


# ── Run three configurations ──────────────────────────────────────────────────
result_nogate = run_alns(
    "No gate",    gate_frac=0.0,            time_budget=args.time, seed=args.seed,
)
result_fixed = run_alns(
    "Fixed gate", gate_frac=args.gate_frac, time_budget=args.time, seed=args.seed,
)
_floor = args.floor_frac  # None → defaults to gate_frac inside the function
result_adaptive = run_alns_adaptive(
    "Adaptive (rolling)", gate_frac=args.gate_frac, time_budget=args.time, seed=args.seed,
    p_explore_init=args.p_explore, p_min=args.p_min, decay=args.p_decay,
    threshold_method="rolling_mean", floor_frac=_floor,
)
result_statistical = run_alns_adaptive(
    f"Adaptive (calib a={args.alpha})", gate_frac=args.gate_frac, time_budget=args.time, seed=args.seed,
    p_explore_init=args.p_explore, p_min=args.p_min, decay=args.p_decay,
    threshold_method="calibration", alpha=args.alpha, floor_frac=_floor,
)

rows = [result_nogate, result_fixed, result_adaptive, result_statistical]


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
    "No gate":                           "#2471A3",
    "Fixed gate":                        "#C0392B",
    "Adaptive (rolling)":                "#27AE60",
    f"Adaptive (calib a={args.alpha})":  "#E67E22",
}
LSTYLE = {
    "No gate":                           "-",
    "Fixed gate":                        "--",
    "Adaptive (rolling)":                "-.",
    f"Adaptive (calib a={args.alpha})":  ":",
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
ra = result_adaptive
rs = result_statistical
table_data = [
    ["",           "No gate",                              "Fixed",                             "Rolling",                         f"Stat(a={args.alpha})"],
    ["Iters",      str(result_nogate["iter_count"]),       str(result_fixed["iter_count"]),     str(ra["iter_count"]),             str(rs["iter_count"])],
    ["Skips",      "—",                                    str(result_fixed["skip_count"]),     str(ra["skip_count"]),             str(rs["skip_count"])],
    ["Skip %",     "—",                                    f"{100*result_fixed['skip_rate']:.0f}%", f"{100*ra['skip_rate']:.0f}%", f"{100*rs['skip_rate']:.0f}%"],
    ["Full evals", str(result_nogate["full_eval_count"]),  str(result_fixed["full_eval_count"]), str(ra["full_eval_count"]),       str(rs["full_eval_count"])],
    ["New bests",  str(result_nogate["new_best_count"]),   str(result_fixed["new_best_count"]),  str(ra["new_best_count"]),        str(rs["new_best_count"])],
    ["Best score", f"{result_nogate['best_score']:.2f}",   f"{result_fixed['best_score']:.2f}", f"{ra['best_score']:.2f}",        f"{rs['best_score']:.2f}"],
]
tbl = ax_tbl.table(
    cellText=table_data[1:], colLabels=table_data[0],
    cellLoc="center", loc="center", bbox=[0.0, 0.05, 1.0, 0.90],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if row == 0:
        cell.set_facecolor("#f0f0f0")
        cell.set_text_props(fontweight="bold")
    elif col == 4 and row > 0:
        cell.set_facecolor("#fef9e7")   # stat gate column
    elif col == 3 and row > 0:
        cell.set_facecolor("#f0faf0")   # rolling adaptive column
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


# ── Plot 2: Convergence — skip rate comparison + p_explore decay ─────────────
WIN = 30

def _rolling(arr, win):
    if len(arr) >= win:
        vals = np.convolve(arr, np.ones(win) / win, mode="valid")
        return np.arange(win, len(arr) + 1), vals
    idx = np.arange(1, len(arr) + 1)
    return idx, np.cumsum(arr) / idx

skip_arr_r = np.array(result_adaptive["skip_per_iter"],    dtype=float)
skip_arr_s = np.array(result_statistical["skip_per_iter"], dtype=float)
p_arr      = np.array(result_adaptive["p_history"],        dtype=float)

ix_r, roll_r = _rolling(skip_arr_r, WIN)
ix_s, roll_s = _rolling(skip_arr_s, WIN)

fig2, (ax_skip, ax_p) = plt.subplots(1, 2, figsize=(13, 4.5))
fig2.suptitle(
    f"Adaptive gate convergence — {Path(args.csv).stem}  "
    f"p_init={args.p_explore}  decay={args.p_decay}  p_min={args.p_min}",
    fontsize=11, y=1.02,
)

ax_skip.plot(ix_r, roll_r * 100, color="#27AE60", linewidth=2.0, label="Rolling-mean gate")
ax_skip.plot(ix_s, roll_s * 100, color="#E67E22", linewidth=2.0, linestyle="--",
             label=f"Statistical gate (a={args.alpha})")
ax_skip.set_xlabel("Iteration", fontsize=11)
ax_skip.set_ylabel("Skip rate — 30-iter rolling avg (%)", fontsize=11)
ax_skip.set_title("Skip rate: rolling vs statistical gate", fontsize=10, pad=6)
ax_skip.grid(axis="y", alpha=0.2, linestyle="--")
ax_skip.set_ylim(0, 105)
ax_skip.legend(fontsize=9, frameon=False)
ax_skip.spines["top"].set_visible(False)
ax_skip.spines["right"].set_visible(False)

ax_p.plot(np.arange(1, len(p_arr) + 1), p_arr * 100, color="#8E44AD", linewidth=2.0)
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

# ── Plot 3: distribution fit, threshold evolution, quantile-vs-α diagnostic ───
# Use the statistical run's ls_log (has same or more data than rolling run)
ls_log_stat   = result_statistical["ls_log"]
ls_log_roll   = result_adaptive["ls_log"]
thr_hist_roll = result_adaptive["threshold_history"]
thr_hist_stat = result_statistical["threshold_history"]
mu_r_final    = result_statistical["mu_r_final"]
sigma_r_final = result_statistical["sigma_r_final"]

ls_log_use = ls_log_stat if len(ls_log_stat) >= len(ls_log_roll) else ls_log_roll

if ls_log_use:
    ratios = np.array([a / b for b, a in ls_log_use if b > 1e-9])

    fig3, (ax_hist, ax_thr, ax_diag) = plt.subplots(1, 3, figsize=(18, 4.5))
    fig3.suptitle(
        f"Statistical gate — fitted distribution & threshold evolution\n"
        f"{Path(args.csv).stem}   N={M}  T={args.T}  n_pairs={len(ratios)}  a={args.alpha}",
        fontsize=11, y=1.02,
    )

    # ── Left: histogram + fitted normal ──────────────────────────────────────
    ax_hist.hist(ratios, bins=25, density=True, color="#27AE60", alpha=0.65,
                 edgecolor="white", label="Observed ratios")
    if not np.isnan(mu_r_final) and sigma_r_final > 1e-9:
        x_fit = np.linspace(ratios.min() - 0.1, ratios.max() + 0.1, 300)
        ax_hist.plot(x_fit, sp_norm.pdf(x_fit, loc=mu_r_final, scale=sigma_r_final),
                     color="#C0392B", linewidth=2.0,
                     label=f"N(mu={mu_r_final:.4f}, s={sigma_r_final:.4f})")
    ax_hist.axvline(float(np.mean(ratios)),   color="#C0392B", linewidth=1.5,
                    linestyle="--", label=f"mean = {np.mean(ratios):.4f}")
    ax_hist.axvline(float(np.median(ratios)), color="#2471A3", linewidth=1.2,
                    linestyle=":",  label=f"median = {np.median(ratios):.4f}")
    ax_hist.axvline(1.0, color="#555", linewidth=1.0, linestyle="-",
                    label="ratio = 1")
    ax_hist.set_xlabel("UE after LS  /  UE before LS", fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.set_title("LS improvement ratio — fitted N(mu, sigma^2)", fontsize=10, pad=6)
    ax_hist.legend(fontsize=8.5, frameon=False)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    # ── Middle: threshold evolution — rolling vs statistical ──────────────────
    proxy_init_val = args.gate_frac * proxy_fn(
        greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=args.seed))
    if thr_hist_roll:
        ax_thr.plot([x[0] for x in thr_hist_roll], [x[1] for x in thr_hist_roll],
                    color="#27AE60", linewidth=2.0, label="Rolling-mean gate")
    if thr_hist_stat:
        ax_thr.plot([x[0] for x in thr_hist_stat], [x[1] for x in thr_hist_stat],
                    color="#E67E22", linewidth=2.0, linestyle="--",
                    label=f"Statistical gate (a={args.alpha})")
    ax_thr.axhline(proxy_init_val, color="#C0392B", linewidth=1.2, linestyle=":",
                   label=f"Fixed gate (frac={args.gate_frac})")
    ax_thr.set_xlabel("Iteration", fontsize=11)
    ax_thr.set_ylabel("proxy_gate value", fontsize=11)
    ax_thr.set_title("Threshold evolution: rolling vs statistical", fontsize=10, pad=6)
    ax_thr.legend(fontsize=8.5, frameon=False)
    ax_thr.spines["top"].set_visible(False)
    ax_thr.spines["right"].set_visible(False)

    # ── Right: calibration regression scatter ────────────────────────────────
    calib_log    = result_statistical.get("all_proxy_ue_log", [])
    b0_f = result_statistical["beta0_final"]
    b1_f = result_statistical["beta1_final"]
    s_f  = result_statistical["sigma_reg_final"]

    if calib_log and not np.isnan(b1_f):
        prx_p = np.array([x[0] for x in calib_log])
        ues_p = np.array([x[1] for x in calib_log])

        # Scatter: mark forced evals (from gate_labels) vs normal-path evals
        gate_lbl  = set(map(tuple, result_statistical["gate_labels"]))
        is_forced = np.array([pt in gate_lbl for pt in calib_log])
        ax_diag.scatter(prx_p[~is_forced], ues_p[~is_forced],
                        color="#27AE60", s=12, alpha=0.5, label="Passed gate")
        ax_diag.scatter(prx_p[is_forced],  ues_p[is_forced],
                        color="#C0392B", s=20, alpha=0.8, marker="x", label="Forced eval (rejected)")

        # Regression line + 1-sigma band
        x_line = np.linspace(prx_p.min(), prx_p.max(), 200)
        y_line = b0_f + b1_f * x_line
        ax_diag.plot(x_line, y_line, color="#8E44AD", linewidth=2.0,
                     label=f"Fit: UE = {b0_f:.1f} + {b1_f:.4f}·proxy")
        ax_diag.fill_between(x_line, y_line - s_f, y_line + s_f,
                             color="#8E44AD", alpha=0.15, label=f"±1σ  (σ={s_f:.3f})")

        # Gate threshold line
        ue_best_final = result_statistical["best_score"]
        z_a = float(sp_norm.ppf(args.alpha))
        thr_proxy = (ue_best_final + s_f * z_a - b0_f) / b1_f
        ax_diag.axvline(thr_proxy, color="#E67E22", linewidth=1.8, linestyle="--",
                        label=f"Gate threshold  proxy={thr_proxy:.1f}")
        ax_diag.axhline(ue_best_final, color="#555", linewidth=1.0, linestyle=":",
                        label=f"UE_best = {ue_best_final:.2f}")

        ax_diag.set_xlabel("Proxy score (before LS)", fontsize=11)
        ax_diag.set_ylabel("UE score (after LS)", fontsize=11)
        ax_diag.set_title(
            f"Calibration: proxy → UE  (a={args.alpha}, n={len(calib_log)})",
            fontsize=10, pad=6)
        ax_diag.legend(fontsize=7.5, frameon=False)
        ax_diag.spines["top"].set_visible(False)
        ax_diag.spines["right"].set_visible(False)
    else:
        ax_diag.text(0.5, 0.5, "Insufficient calibration data",
                     ha="center", va="center", transform=ax_diag.transAxes, fontsize=11)
        ax_diag.axis("off")

    plt.tight_layout()
    ratio_path = OUT_DIR / "adaptive_ls_ratios.png"
    fig3.savefig(ratio_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig3)
    plt.rcdefaults()
    print(f"Ratio plot       -> {ratio_path}")

    ratio_csv = OUT_DIR / "ls_improvement_log.csv"
    pd.DataFrame(ls_log_use, columns=["ue_before_ls", "ue_after_ls"]).assign(
        ratio=ratios
    ).to_csv(ratio_csv, index=False)
    print(f"LS pairs CSV     -> {ratio_csv}")
else:
    print("No LS pairs collected (adaptive run ended before any normal LS calls).")

ue_eval.dispose()
print("Done.")
