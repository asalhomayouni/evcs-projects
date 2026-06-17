"""
alpha_sweep.py  —  Sweep alpha values for the calibration gate
Runs: no-gate / fixed-gate / rolling (once each as baselines),
      then calibration with each alpha in ALPHA_VALUES, all floor=0.90.

Usage:
    python scripts/alpha_sweep.py
    python scripts/alpha_sweep.py --time 60 --out-tag mysweep
"""
import argparse, sys, time
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
p.add_argument("--csv",         default="center_79_Monza_k400.csv")
p.add_argument("--T",           type=int,   default=6)
p.add_argument("--D",           type=float, default=2.0)
p.add_argument("--seed",        type=int,   default=11)
p.add_argument("--time",        type=float, default=90.0)
p.add_argument("--gate-frac",   type=float, default=0.90)
p.add_argument("--floor-frac",  type=float, default=0.90)
p.add_argument("--ls-moves",    type=int,   default=8)
p.add_argument("--frac-remove", type=float, default=0.20)
p.add_argument("--ue-alpha-wait",       type=float, default=10.0)
p.add_argument("--ue-noopt",            type=float, default=10.0)
p.add_argument("--ue-penalty-distance", type=float, default=5.0)
p.add_argument("--ue-cap-per-server",   type=float, default=4.0)
p.add_argument("--ue-max-range",        type=float, default=2.0)
p.add_argument("--p-explore",   type=float, default=0.30)
p.add_argument("--p-min",       type=float, default=0.02)
p.add_argument("--p-decay",     type=float, default=0.995)
p.add_argument("--out-tag",     type=str,   default=None)
args = p.parse_args()

# Alpha values to sweep — spans from very lenient to equivalent-to-rolling
ALPHA_VALUES = [0.0001, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50]

import datetime as _dt
_tag = args.out_tag or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / "gate" / "alpha_sweep" / _tag
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHARGERS  = 6
Q_CAP         = 20.0
CUMULATIVE    = True
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]
LS_MODES      = ("site_swap", "local_remove")

# ── Instance ──────────────────────────────────────────────────────────────────
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

print(f"Instance : {args.csv}   N={M}  T={args.T}  time={args.time}s")
print(f"Alpha sweep: {ALPHA_VALUES}")

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


# ── Shared ALNS core (all methods call this) ──────────────────────────────────
N_WARMUP       = 10
EMA_WINDOW     = 30
N_WARMUP_CALIB = 20
CALIB_WINDOW   = 100

def run_alns(label, gate_frac, time_budget, seed,
             p_explore_init=0.30, p_min=0.02, decay=0.995,
             threshold_method="fixed", alpha=0.10, floor_frac=None):

    rng = np.random.default_rng(seed + 100)
    U_curr     = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=seed)
    score_curr = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    proxy_init  = proxy_fn(U_curr)
    proxy_gate  = gate_frac * proxy_init if gate_frac > 0 else 0.0
    _floor_frac = floor_frac if floor_frac is not None else gate_frac
    proxy_floor = _floor_frac * proxy_init

    ls_log           = []
    all_proxy_ue_log = []
    avg_improvement  = 1.0
    p_explore        = p_explore_init

    t_vec, score_vec = [0.0], [score_best]
    skip_count = forced_eval_count = full_eval_count = iter_count = new_best_count = 0

    print(f"\n  [{label}]", flush=True)
    t0 = time.perf_counter()

    while True:
        if time.perf_counter() - t0 >= time_budget:
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

        p_val = proxy_fn(U_recon) if proxy_gate > 0 else None

        gate_rejected = False
        if proxy_gate > 0 and p_val < proxy_gate:
            gate_rejected = True
            if rng.random() < p_explore:
                ue_score = ue_fn(U_recon)
                all_proxy_ue_log.append((p_val, ue_score))
                forced_eval_count += 1
                full_eval_count   += 1
            else:
                skip_count += 1

        p_explore = max(p_min, p_explore * decay)

        if gate_rejected:
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

        ue_before = float(ue_scores[0])
        if ue_before > 1e-9:
            ls_log.append((ue_before, float(iter_score)))
        if p_val is not None:
            all_proxy_ue_log.append((p_val, iter_score))

        # ── Update adaptive threshold ─────────────────────────────────────────
        enough_ls    = len(ls_log) >= N_WARMUP
        enough_calib = len(all_proxy_ue_log) >= N_WARMUP_CALIB
        can_update   = proxy_gate > 0 and (
            (threshold_method == "calibration" and enough_calib) or
            (threshold_method == "rolling_mean" and enough_ls)
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
                        z_alpha   = float(sp_norm.ppf(alpha))
                        proxy_gate_adapt = (score_best + sigma_reg * z_alpha - beta0) / beta1
                        avg_improvement  = proxy_gate_adapt
                        use_fallback = False
                if use_fallback:
                    ratios_w = [a / b for b, a in ls_log[-EMA_WINDOW:] if b > 1e-9]
                    avg_improvement  = float(np.mean(ratios_w)) if ratios_w else 1.0
                    proxy_gate_adapt = proxy_fn(U_best) / avg_improvement
            else:
                ratios_w = [a / b for b, a in ls_log[-EMA_WINDOW:] if b > 1e-9]
                avg_improvement  = float(np.mean(ratios_w)) if ratios_w else 1.0
                proxy_gate_adapt = proxy_fn(U_best) / avg_improvement

            proxy_gate = min(proxy_gate_adapt, proxy_floor)

        if iter_score >= score_curr:
            U_curr, score_curr = dict(iter_U), iter_score
        if iter_score > score_best:
            U_best, score_best = dict(iter_U), iter_score
            new_best_count += 1

        elapsed = time.perf_counter() - t0
        t_vec.append(elapsed)
        score_vec.append(score_best)

    total_time = time.perf_counter() - t0
    skip_rate  = skip_count / max(iter_count, 1)
    print(f"    iters={iter_count}  skips={skip_count} ({100*skip_rate:.1f}%)  "
          f"full_evals={full_eval_count}  new_bests={new_best_count}  "
          f"best={score_best:.3f}", flush=True)

    return {
        "label":           label,
        "alpha":           alpha,
        "threshold_method": threshold_method,
        "iter_count":      iter_count,
        "skip_count":      skip_count,
        "skip_rate":       skip_rate,
        "full_eval_count": full_eval_count,
        "new_best_count":  new_best_count,
        "best_score":      score_best,
        "t_vec":           t_vec,
        "score_vec":       score_vec,
        "calib_pairs":     len(all_proxy_ue_log),
    }


# ── Run baselines (once) ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("  BASELINES")
print("="*60)
r_nogate  = run_alns("No gate",    gate_frac=0.0,           time_budget=args.time, seed=args.seed)
r_fixed   = run_alns("Fixed gate", gate_frac=args.gate_frac, time_budget=args.time, seed=args.seed,
                      threshold_method="fixed")
r_rolling = run_alns("Rolling (floor=0.90)", gate_frac=args.gate_frac,
                      time_budget=args.time, seed=args.seed,
                      p_explore_init=args.p_explore, p_min=args.p_min, decay=args.p_decay,
                      threshold_method="rolling_mean", floor_frac=args.floor_frac)

nogate_score  = r_nogate["best_score"]
rolling_score = r_rolling["best_score"]

# ── Run calibration for each alpha ────────────────────────────────────────────
print("\n" + "="*60)
print("  ALPHA SWEEP")
print("="*60)
calib_results = []
for a in ALPHA_VALUES:
    r = run_alns(
        f"Calib a={a}", gate_frac=args.gate_frac, time_budget=args.time, seed=args.seed,
        p_explore_init=args.p_explore, p_min=args.p_min, decay=args.p_decay,
        threshold_method="calibration", alpha=a, floor_frac=args.floor_frac,
    )
    calib_results.append(r)

# ── Summary table ─────────────────────────────────────────────────────────────
all_results = [r_nogate, r_fixed, r_rolling] + calib_results

rows = []
for r in all_results:
    rows.append({
        "config":          r["label"],
        "alpha":           r["alpha"],
        "iterations":      r["iter_count"],
        "skip_rate_%":     round(100 * r["skip_rate"], 1),
        "full_evals":      r["full_eval_count"],
        "new_bests":       r["new_best_count"],
        "best_score":      round(r["best_score"], 3),
        "gain_vs_nogate":  round(r["best_score"] - nogate_score, 3),
        "delta_vs_rolling": round(r["best_score"] - rolling_score, 3),
    })

df = pd.DataFrame(rows)
print("\n" + "="*72)
print(df.to_string(index=False))
print("="*72)

csv_path = OUT_DIR / "alpha_sweep_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nCSV  -> {csv_path}")


# ── Plot 1: alpha vs best score + delta vs rolling ────────────────────────────
alphas = [r["alpha"] for r in calib_results]
scores = [r["best_score"] for r in calib_results]
deltas = [r["best_score"] - rolling_score for r in calib_results]
skips  = [100 * r["skip_rate"] for r in calib_results]

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle(
    f"Alpha sweep — calibration gate   {Path(args.csv).stem}   "
    f"N={M}  T={args.T}  floor={args.floor_frac}  time={args.time}s",
    fontsize=11, y=1.02,
)

# ── Left: best score vs alpha ──────────────────────────────────────────────
ax = axes[0]
ax.semilogx(alphas, scores, "o-", color="#E67E22", linewidth=2, markersize=7, zorder=3)
ax.axhline(rolling_score, color="#27AE60", linewidth=1.8, linestyle="--",
           label=f"Rolling (floor=0.90) = {rolling_score:.3f}")
ax.axhline(nogate_score,  color="#2471A3", linewidth=1.2, linestyle=":",
           label=f"No gate = {nogate_score:.3f}")
# Mark best alpha
best_idx = int(np.argmax(scores))
ax.scatter([alphas[best_idx]], [scores[best_idx]], color="#C0392B", s=120,
           zorder=5, label=f"Best: α={alphas[best_idx]}  score={scores[best_idx]:.3f}")
ax.set_xlabel("α  (log scale)", fontsize=11)
ax.set_ylabel("Best UE score", fontsize=11)
ax.set_title("Best score vs α", fontsize=10, pad=6)
ax.legend(fontsize=8.5, frameon=False)
ax.grid(axis="y", alpha=0.2, linestyle="--")

# ── Middle: delta vs rolling ───────────────────────────────────────────────
ax2 = axes[1]
colors_d = ["#27AE60" if d > 0.01 else ("#F39C12" if abs(d) <= 0.01 else "#C0392B")
            for d in deltas]
bars = ax2.bar(range(len(alphas)), deltas, color=colors_d, edgecolor="white", linewidth=0.8)
ax2.axhline(0, color="#555", linewidth=1.2, linestyle="-")
ax2.set_xticks(range(len(alphas)))
ax2.set_xticklabels([str(a) for a in alphas], rotation=40, ha="right", fontsize=8.5)
ax2.set_xlabel("α", fontsize=11)
ax2.set_ylabel("Δ best score vs rolling", fontsize=11)
ax2.set_title("Calibration gain over rolling\n(green = better, red = worse)", fontsize=10, pad=6)
ax2.grid(axis="y", alpha=0.2, linestyle="--")
for bar, d in zip(bars, deltas):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + (0.02 if d >= 0 else -0.06),
             f"{d:+.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

# ── Right: skip rate vs alpha ──────────────────────────────────────────────
ax3 = axes[2]
ax3.semilogx(alphas, skips, "s-", color="#8E44AD", linewidth=2, markersize=7)
ax3.axhline(100 * r_rolling["skip_rate"], color="#27AE60", linewidth=1.5,
            linestyle="--", label=f"Rolling skip rate {100*r_rolling['skip_rate']:.1f}%")
ax3.set_xlabel("α  (log scale)", fontsize=11)
ax3.set_ylabel("Skip rate (%)", fontsize=11)
ax3.set_title("Skip rate vs α\n(higher α → stricter gate → more skips)", fontsize=10, pad=6)
ax3.legend(fontsize=9, frameon=False)
ax3.grid(axis="y", alpha=0.2, linestyle="--")

plt.tight_layout()
plot_path = OUT_DIR / "alpha_sweep_plot.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Plot -> {plot_path}")

# ── Plot 2: trajectory curves for all alphas + baselines ─────────────────────
fig2, ax_t = plt.subplots(figsize=(13, 5.5))
fig2.suptitle(
    f"Score trajectories — all α values   {Path(args.csv).stem}",
    fontsize=11, y=1.02,
)

# Baselines
ax_t.plot(r_nogate["t_vec"],  r_nogate["score_vec"],  color="#2471A3",
          linewidth=1.5, linestyle=":", label=f"No gate  {r_nogate['best_score']:.2f}")
ax_t.plot(r_fixed["t_vec"],   r_fixed["score_vec"],   color="#C0392B",
          linewidth=1.5, linestyle="--", label=f"Fixed gate  {r_fixed['best_score']:.2f}")
ax_t.plot(r_rolling["t_vec"], r_rolling["score_vec"], color="#27AE60",
          linewidth=2.5, linestyle="-.", label=f"Rolling  {r_rolling['best_score']:.2f}")

# Calib curves — use a colour gradient from light to dark orange
cmap = plt.cm.plasma
for i, r in enumerate(calib_results):
    c = cmap(0.15 + 0.7 * i / max(len(calib_results) - 1, 1))
    ax_t.plot(r["t_vec"], r["score_vec"], color=c, linewidth=1.4, alpha=0.85,
              label=f"α={r['alpha']}  {r['best_score']:.2f}")

ax_t.set_xlabel("Wall-clock time (s)", fontsize=11)
ax_t.set_ylabel("UE score (demand served)", fontsize=11)
ax_t.set_xlim(0, args.time)
ax_t.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower right")
ax_t.grid(axis="y", alpha=0.2, linestyle="--")
ax_t.spines["top"].set_visible(False)
ax_t.spines["right"].set_visible(False)

plt.tight_layout()
traj_path = OUT_DIR / "alpha_sweep_trajectories.png"
fig2.savefig(traj_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig2)
plt.rcdefaults()
print(f"Trajectories -> {traj_path}")

ue_eval.dispose()
print("\nDone.")
