"""
multi_seed_benchmark.py
Runs no-gate / fixed-gate / rolling / calibration across multiple seeds
and reports mean +/- std to give a statistically reliable comparison.

Usage:
    python scripts/multi_seed_benchmark.py
    python scripts/multi_seed_benchmark.py --time 90 --seeds 11 22 33 44 55
"""
import argparse, sys, time, io
# Force UTF-8 stdout so Unicode characters print on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm as sp_norm, ttest_rel, wilcoxon

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
p.add_argument("--time",        type=float, default=90.0)
p.add_argument("--alpha",       type=float, default=0.01,
               help="Alpha for calibration gate (best from sweep)")
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
p.add_argument("--seeds",       type=int,   nargs="+", default=[11, 22, 33, 44, 55])
p.add_argument("--out-tag",     type=str,   default=None)
args = p.parse_args()

import datetime as _dt
_tag = args.out_tag or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / "gate" / "multi_seed" / _tag
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["no_gate", "fixed_gate", "rolling", "calibration"]
METHOD_LABELS = {
    "no_gate":     "No gate",
    "fixed_gate":  "Fixed gate",
    "rolling":     f"Rolling\n(floor={args.floor_frac})",
    "calibration": f"Calib\n(a={args.alpha}, floor={args.floor_frac})",
}
MAX_CHARGERS  = 6
Q_CAP         = 20.0
CUMULATIVE    = True
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]
LS_MODES      = ("site_swap", "local_remove")
N_WARMUP       = 10
EMA_WINDOW     = 30
N_WARMUP_CALIB = 20
CALIB_WINDOW   = 100

# ── Build instance (fixed — same for all seeds) ───────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "input"
df_raw    = pd.read_csv(DATA_DIR / args.csv)
coords_km = df_raw[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float) * 111.0
pop       = np.maximum(df_raw["Aggregated_Population"].to_numpy(float), 0.0)
M         = coords_km.shape[0]
P_T       = [max(4, M // 30)] * args.T

# Demand is seeded from seed=11 always (instance is fixed)
rng_inst  = np.random.default_rng(11)
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
print(f"Seeds    : {args.seeds}")
print(f"Alpha    : {args.alpha}   floor_frac: {args.floor_frac}")
print(f"Total runs: {len(args.seeds)} seeds × {len(METHODS)} methods = "
      f"{len(args.seeds)*len(METHODS)} runs  (~{len(args.seeds)*len(METHODS)*args.time/60:.0f} min)\n")

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


# ── Single run function ───────────────────────────────────────────────────────
def run_single(method: str, seed: int) -> dict:
    gate_frac   = args.gate_frac if method != "no_gate" else 0.0
    floor_frac  = args.floor_frac
    alpha       = args.alpha
    p_explore_i = args.p_explore if method in ("rolling", "calibration") else 0.0
    p_min       = args.p_min
    decay       = args.p_decay

    rng = np.random.default_rng(seed + 100)
    U_curr     = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=seed)
    score_curr = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    proxy_init  = proxy_fn(U_curr)
    proxy_gate  = gate_frac * proxy_init if gate_frac > 0 else 0.0
    proxy_floor = floor_frac * proxy_init

    ls_log           = []
    all_proxy_ue_log = []
    avg_improvement  = 1.0
    p_explore        = p_explore_i

    t_vec, score_vec = [0.0], [score_best]
    skip_count = forced_count = full_evals = iter_count = new_bests = 0

    t0 = time.perf_counter()
    while True:
        if time.perf_counter() - t0 >= args.time:
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
                ue_s = ue_fn(U_recon)
                all_proxy_ue_log.append((p_val, ue_s))
                forced_count += 1
                full_evals   += 1
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
        full_evals += len(visited)
        best_idx    = int(np.argmax(ue_scores))
        iter_score  = float(ue_scores[best_idx])
        iter_U      = visited[best_idx]

        ue_before = float(ue_scores[0])
        if ue_before > 1e-9:
            ls_log.append((ue_before, float(iter_score)))
        if p_val is not None:
            all_proxy_ue_log.append((p_val, iter_score))

        # Update threshold
        enough_ls    = len(ls_log) >= N_WARMUP
        enough_calib = len(all_proxy_ue_log) >= N_WARMUP_CALIB
        can_update   = proxy_gate > 0 and method in ("rolling", "calibration") and (
            (method == "calibration" and enough_calib) or
            (method == "rolling"     and enough_ls)
        )
        if can_update:
            if method == "calibration":
                wc = all_proxy_ue_log[-CALIB_WINDOW:]
                prx = np.array([x[0] for x in wc])
                ues = np.array([x[1] for x in wc])
                adapt = None
                if np.std(prx) > 1e-9 and len(prx) >= 4:
                    b1, b0 = np.polyfit(prx, ues, 1)
                    if b1 > 1e-9:
                        res      = ues - (b0 + b1 * prx)
                        sigma_r  = max(float(np.std(res, ddof=2)), 1e-9)
                        z_a      = float(sp_norm.ppf(alpha))
                        adapt    = (score_best + sigma_r * z_a - b0) / b1
                if adapt is None:
                    ratios = [a / b for b, a in ls_log[-EMA_WINDOW:] if b > 1e-9]
                    adapt  = proxy_fn(U_best) / (np.mean(ratios) if ratios else 1.0)
            else:
                ratios = [a / b for b, a in ls_log[-EMA_WINDOW:] if b > 1e-9]
                adapt  = proxy_fn(U_best) / (np.mean(ratios) if ratios else 1.0)
            proxy_gate = min(adapt, proxy_floor)

        if iter_score >= score_curr:
            U_curr, score_curr = dict(iter_U), iter_score
        if iter_score > score_best:
            U_best, score_best = dict(iter_U), iter_score
            new_bests += 1

        elapsed = time.perf_counter() - t0
        t_vec.append(elapsed)
        score_vec.append(score_best)

    return {
        "method":      method,
        "seed":        seed,
        "best_score":  score_best,
        "skip_rate":   skip_count / max(iter_count, 1),
        "full_evals":  full_evals,
        "new_bests":   new_bests,
        "iter_count":  iter_count,
        "t_vec":       t_vec,
        "score_vec":   score_vec,
    }


# ── Run all seeds × methods ───────────────────────────────────────────────────
all_results = {m: [] for m in METHODS}
total = len(args.seeds) * len(METHODS)
done  = 0

for seed in args.seeds:
    print(f"\n{'='*60}")
    print(f"  SEED {seed}")
    print(f"{'='*60}")
    for method in METHODS:
        done += 1
        print(f"  [{done}/{total}]  {method}  seed={seed}", flush=True)
        r = run_single(method, seed)
        all_results[method].append(r)
        print(f"    best={r['best_score']:.3f}  skip={100*r['skip_rate']:.1f}%  "
              f"evals={r['full_evals']}  new_bests={r['new_bests']}", flush=True)


# ── Statistics table ──────────────────────────────────────────────────────────
print(f"\n{'='*72}")
print(f"  RESULTS SUMMARY  ({len(args.seeds)} seeds)")
print(f"{'='*72}")

stats_rows = []
for method in METHODS:
    scores = [r["best_score"]  for r in all_results[method]]
    skips  = [100*r["skip_rate"] for r in all_results[method]]
    evals  = [r["full_evals"]  for r in all_results[method]]
    stats_rows.append({
        "method":      METHOD_LABELS[method].replace("\n", " "),
        "mean_score":  round(float(np.mean(scores)), 3),
        "std_score":   round(float(np.std(scores, ddof=1)), 3),
        "min_score":   round(float(np.min(scores)), 3),
        "max_score":   round(float(np.max(scores)), 3),
        "mean_skip_%": round(float(np.mean(skips)),  1),
        "mean_evals":  round(float(np.mean(evals)),  0),
        **{f"seed_{s}": round(scores[i], 3)
           for i, s in enumerate(args.seeds)},
    })

df_stats = pd.DataFrame(stats_rows)
print(df_stats[["method","mean_score","std_score","min_score","max_score",
                "mean_skip_%","mean_evals"]].to_string(index=False))

# Statistical tests (calibration vs rolling)
calib_scores   = [r["best_score"] for r in all_results["calibration"]]
rolling_scores = [r["best_score"] for r in all_results["rolling"]]
nogate_scores  = [r["best_score"] for r in all_results["no_gate"]]

print(f"\n  Per-seed  Calibration vs Rolling:")
for s, cs, rs in zip(args.seeds, calib_scores, rolling_scores):
    diff = cs - rs
    marker = "WIN" if diff > 0 else ("TIE" if abs(diff) < 0.01 else "LOSS")
    print(f"    seed={s}  calib={cs:.3f}  rolling={rs:.3f}  delta={diff:+.3f}  {marker}")

mean_diff = np.mean([c - r for c, r in zip(calib_scores, rolling_scores)])
print(f"\n  Mean delta (calib - rolling) = {mean_diff:+.3f}")

if len(args.seeds) >= 5:
    _, p_ttest = ttest_rel(calib_scores, rolling_scores)
    try:
        _, p_wilcox = wilcoxon(calib_scores, rolling_scores)
        print(f"  Paired t-test  p = {p_ttest:.4f}  "
              f"({'significant' if p_ttest < 0.05 else 'not significant'} at 5%)")
        print(f"  Wilcoxon test  p = {p_wilcox:.4f}  "
              f"({'significant' if p_wilcox < 0.05 else 'not significant'} at 5%)")
    except Exception:
        print(f"  Paired t-test  p = {p_ttest:.4f}")

csv_path = OUT_DIR / "multi_seed_results.csv"
df_stats.to_csv(csv_path, index=False)

# Full per-seed CSV
per_seed_rows = []
for method in METHODS:
    for r in all_results[method]:
        nogate_s = next(x["best_score"] for x in all_results["no_gate"]
                        if x["seed"] == r["seed"])
        per_seed_rows.append({
            "method":          METHOD_LABELS[method].replace("\n", " "),
            "seed":            r["seed"],
            "best_score":      r["best_score"],
            "gain_vs_nogate":  round(r["best_score"] - nogate_s, 3),
            "skip_rate_%":     round(100 * r["skip_rate"], 1),
            "full_evals":      r["full_evals"],
            "new_bests":       r["new_bests"],
        })
df_per_seed = pd.DataFrame(per_seed_rows)
per_seed_csv = OUT_DIR / "multi_seed_per_seed.csv"
df_per_seed.to_csv(per_seed_csv, index=False)
print(f"\nCSV  -> {csv_path}")
print(f"CSV  -> {per_seed_csv}")


# ── Plot 1: Box plot + individual points ──────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

COLORS_M = {
    "no_gate":     "#2471A3",
    "fixed_gate":  "#C0392B",
    "rolling":     "#27AE60",
    "calibration": "#E67E22",
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    f"Multi-seed benchmark — {Path(args.csv).stem}   N={M}  T={args.T}  "
    f"time={args.time}s   seeds={args.seeds}",
    fontsize=11, y=1.02,
)

# ── Left: box + strip ─────────────────────────────────────────────────────────
ax = axes[0]
positions = np.arange(len(METHODS))
for xi, method in enumerate(METHODS):
    scores = [r["best_score"] for r in all_results[method]]
    c = COLORS_M[method]
    bp = ax.boxplot(scores, positions=[xi], widths=0.45,
                    patch_artist=True, manage_ticks=False,
                    boxprops=dict(facecolor=c, alpha=0.45, linewidth=1.2),
                    medianprops=dict(color="black", linewidth=2.0),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=5))
    jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(scores))
    ax.scatter([xi + j for j in jitter], scores, color=c, s=55,
               zorder=4, edgecolors="white", linewidths=0.8)
    ax.text(xi, float(np.mean(scores)) - 0.18,
            f"μ={np.mean(scores):.2f}", ha="center", fontsize=8.5,
            color=c, fontweight="bold")

ax.set_xticks(positions)
ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=9)
ax.set_ylabel("Best UE score", fontsize=11)
ax.set_title("Score distribution across seeds\n(box = IQR, line = median, dots = seeds)",
             fontsize=10, pad=6)
ax.grid(axis="y", alpha=0.2, linestyle="--")

# ── Right: mean ± std bar chart with per-seed lines ───────────────────────────
ax2 = axes[1]
means = [np.mean([r["best_score"] for r in all_results[m]]) for m in METHODS]
stds  = [np.std([r["best_score"]  for r in all_results[m]], ddof=1) for m in METHODS]

bars = ax2.bar(positions, means, yerr=stds, capsize=6,
               color=[COLORS_M[m] for m in METHODS],
               alpha=0.7, edgecolor="white", linewidth=0.8,
               error_kw=dict(elinewidth=1.5, ecolor="black"))

# Add lines connecting same seed across methods for visual pairing
for si in range(len(args.seeds)):
    seed_scores = [all_results[m][si]["best_score"] for m in METHODS]
    ax2.plot(positions, seed_scores, "o-", color="#888", alpha=0.35,
             linewidth=1.0, markersize=4)

ax2.set_xticks(positions)
ax2.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=9)
ax2.set_ylabel("Mean best UE score", fontsize=11)
ax2.set_title(f"Mean +/- std  ({len(args.seeds)} seeds)\n"
              f"Lines connect same seed across methods",
              fontsize=10, pad=6)
ax2.grid(axis="y", alpha=0.2, linestyle="--")

# Annotate mean values
ymin = ax2.get_ylim()[0]
for xi, (m, mu, sd) in enumerate(zip(METHODS, means, stds)):
    ax2.text(xi, mu + sd + 0.05, f"{mu:.2f}\n+/-{sd:.2f}",
             ha="center", va="bottom", fontsize=8, color=COLORS_M[m],
             fontweight="bold")

plt.tight_layout()
box_path = OUT_DIR / "multi_seed_boxplot.png"
fig.savefig(box_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Plot -> {box_path}")


# ── Plot 2: trajectories per method (all seeds) ───────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(11, 9), sharey=True)
axes2 = axes2.flatten()
fig2.suptitle(
    f"Score trajectories by method — {len(args.seeds)} seeds each",
    fontsize=11, y=1.02,
)

for ax_i, method in enumerate(METHODS):
    ax_m = axes2[ax_i]
    c = COLORS_M[method]
    seed_scores_at_t = []
    for r in all_results[method]:
        t_interp = np.linspace(0, args.time, 200)
        s_interp = np.interp(t_interp, r["t_vec"], r["score_vec"])
        seed_scores_at_t.append(s_interp)
        ax_m.plot(r["t_vec"], r["score_vec"], color=c, alpha=0.3, linewidth=1.2)

    # Mean trajectory
    mean_traj = np.mean(seed_scores_at_t, axis=0)
    t_interp  = np.linspace(0, args.time, 200)
    ax_m.plot(t_interp, mean_traj, color=c, linewidth=2.5,
              label=f"Mean  {np.mean([r['best_score'] for r in all_results[method]]):.2f}")

    ax_m.set_title(METHOD_LABELS[method], fontsize=10, pad=6)
    ax_m.set_xlabel("Time (s)", fontsize=10)
    if ax_i % 2 == 0:
        ax_m.set_ylabel("UE score", fontsize=10)
    ax_m.legend(fontsize=9, frameon=False)
    ax_m.grid(axis="y", alpha=0.2, linestyle="--")
    ax_m.spines["top"].set_visible(False)
    ax_m.spines["right"].set_visible(False)
    ax_m.set_xlim(0, args.time)

plt.tight_layout()
traj_path = OUT_DIR / "multi_seed_trajectories.png"
fig2.savefig(traj_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig2)
plt.rcdefaults()
print(f"Plot -> {traj_path}")

ue_eval.dispose()
print("\nDone.")
