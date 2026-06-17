"""
alpha_sweep_multiseed.py
Run rolling + calibration(α) across multiple seeds.
For each α: report mean best score ± std and mean skip rate ± std.
Goal: find the α with best average quality at a reasonable skip rate (20-30%).

Usage:
    python scripts/alpha_sweep_multiseed.py
    python scripts/alpha_sweep_multiseed.py --time 90 --seeds 11 22 33 44 55

Runtime estimate: len(seeds) × (1 rolling + len(alphas)) × time
    e.g. 5 × 11 × 90s ≈ 82 min
"""
import argparse, sys, time, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
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
p.add_argument("--seeds",       type=int,   nargs="+", default=[11, 22, 33, 44, 55])
p.add_argument("--out-tag",     type=str,   default=None)
args = p.parse_args()

# Alpha values to sweep
ALPHA_VALUES = [0.0001, 0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.10, 0.20, 0.50]

import datetime as _dt
_tag = args.out_tag or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / "gate" / "alpha_sweep_multiseed" / _tag
OUT_DIR.mkdir(parents=True, exist_ok=True)

total_runs = len(args.seeds) * (1 + len(ALPHA_VALUES))
print(f"Instance : {args.csv}   N=?, T={args.T}  time={args.time}s")
print(f"Seeds    : {args.seeds}  ({len(args.seeds)} seeds)")
print(f"Alphas   : {ALPHA_VALUES}")
print(f"Total runs: {total_runs}  (~{total_runs * args.time / 60:.0f} min)\n")

MAX_CHARGERS  = 6
Q_CAP         = 20.0
CUMULATIVE    = True
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]
LS_MODES      = ("site_swap", "local_remove")
N_WARMUP       = 10
EMA_WINDOW     = 30
N_WARMUP_CALIB = 20
CALIB_WINDOW   = 100

# ── Build instance ────────────────────────────────────────────────────────────
DATA_DIR  = PROJECT_ROOT / "data" / "input"
df_raw    = pd.read_csv(DATA_DIR / args.csv)
coords_km = df_raw[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float) * 111.0
pop       = np.maximum(df_raw["Aggregated_Population"].to_numpy(float), 0.0)
M         = coords_km.shape[0]
P_T       = [max(4, M // 30)] * args.T

rng_inst  = np.random.default_rng(11)   # instance fixed
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

print(f"N={M}  T={args.T}")

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


# ── Core ALNS runner ──────────────────────────────────────────────────────────
def run_single(method: str, seed: int, alpha: float = 0.01) -> dict:
    gate_frac  = args.gate_frac
    floor_frac = args.floor_frac

    rng = np.random.default_rng(seed + 100)
    U_curr     = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=seed)
    score_curr = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    proxy_init  = proxy_fn(U_curr)
    proxy_gate  = gate_frac * proxy_init
    proxy_floor = floor_frac * proxy_init

    ls_log           = []
    all_proxy_ue_log = []
    p_explore        = args.p_explore

    skip_count = full_evals = iter_count = new_bests = 0

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

        p_val = proxy_fn(U_recon)
        gate_rejected = False
        if p_val < proxy_gate:
            gate_rejected = True
            if rng.random() < p_explore:
                ue_s = ue_fn(U_recon)
                all_proxy_ue_log.append((p_val, ue_s))
                full_evals += 1
            else:
                skip_count += 1

        p_explore = max(args.p_min, p_explore * args.p_decay)
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

        if float(ue_scores[0]) > 1e-9:
            ls_log.append((float(ue_scores[0]), iter_score))
        if p_val is not None:
            all_proxy_ue_log.append((p_val, iter_score))

        # ── Threshold update ──────────────────────────────────────────────────
        enough_ls    = len(ls_log) >= N_WARMUP
        enough_calib = len(all_proxy_ue_log) >= N_WARMUP_CALIB

        if (method == "calibration" and enough_calib) or \
           (method == "rolling"     and enough_ls):
            if method == "calibration":
                wc = all_proxy_ue_log[-CALIB_WINDOW:]
                prx = np.array([x[0] for x in wc])
                ues = np.array([x[1] for x in wc])
                adapt = None
                if np.std(prx) > 1e-9 and len(prx) >= 4:
                    b1, b0 = np.polyfit(prx, ues, 1)
                    if b1 > 1e-9:
                        res     = ues - (b0 + b1 * prx)
                        sig     = max(float(np.std(res, ddof=2)), 1e-9)
                        adapt   = (score_best + sig * float(sp_norm.ppf(alpha)) - b0) / b1
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

    return {
        "method":     method,
        "alpha":      alpha,
        "seed":       seed,
        "best_score": score_best,
        "skip_rate":  skip_count / max(iter_count, 1),
        "full_evals": full_evals,
        "new_bests":  new_bests,
    }


# ── Run all seeds × (rolling + calibration × alphas) ─────────────────────────
records  = []
run_num  = 0

for seed in args.seeds:
    print(f"\n{'='*60}  SEED {seed}  {'='*10}")

    # Rolling baseline for this seed
    run_num += 1
    print(f"  [{run_num}/{total_runs}]  rolling  seed={seed}", flush=True)
    r = run_single("rolling", seed, alpha=0.0)
    records.append(r)
    print(f"    best={r['best_score']:.3f}  skip={100*r['skip_rate']:.1f}%", flush=True)

    # Calibration sweep
    for alpha in ALPHA_VALUES:
        run_num += 1
        print(f"  [{run_num}/{total_runs}]  calib a={alpha}  seed={seed}", flush=True)
        r = run_single("calibration", seed, alpha=alpha)
        records.append(r)
        print(f"    best={r['best_score']:.3f}  skip={100*r['skip_rate']:.1f}%", flush=True)

df_all = pd.DataFrame(records)

# ── Aggregate by (method, alpha) ──────────────────────────────────────────────
rows_agg = []
rolling_mean = df_all[df_all["method"] == "rolling"]["best_score"].mean()

for alpha in ALPHA_VALUES:
    sub = df_all[(df_all["method"] == "calibration") & (df_all["alpha"] == alpha)]
    mean_s  = sub["best_score"].mean()
    std_s   = sub["best_score"].std(ddof=1)
    mean_sk = sub["skip_rate"].mean() * 100
    std_sk  = sub["skip_rate"].std(ddof=1) * 100
    mean_e  = sub["full_evals"].mean()
    delta   = mean_s - rolling_mean
    rows_agg.append({
        "alpha":           alpha,
        "mean_score":      round(mean_s,  3),
        "std_score":       round(std_s,   3),
        "mean_skip_%":     round(mean_sk, 1),
        "std_skip_%":      round(std_sk,  1),
        "mean_full_evals": round(mean_e,  0),
        "delta_vs_rolling":round(delta,   3),
    })

# Rolling row
roll_sub = df_all[df_all["method"] == "rolling"]
rows_agg.insert(0, {
    "alpha":           "rolling",
    "mean_score":      round(rolling_mean, 3),
    "std_score":       round(roll_sub["best_score"].std(ddof=1), 3),
    "mean_skip_%":     round(roll_sub["skip_rate"].mean()*100, 1),
    "std_skip_%":      round(roll_sub["skip_rate"].std(ddof=1)*100, 1),
    "mean_full_evals": round(roll_sub["full_evals"].mean(), 0),
    "delta_vs_rolling":0.0,
})

df_agg = pd.DataFrame(rows_agg)
print(f"\n{'='*80}")
print(f"ALPHA SWEEP SUMMARY  ({len(args.seeds)} seeds each)")
print(f"Rolling baseline mean = {rolling_mean:.3f}")
print(f"{'='*80}")
print(df_agg.to_string(index=False))

csv_agg  = OUT_DIR / "alpha_sweep_multiseed_summary.csv"
csv_full = OUT_DIR / "alpha_sweep_multiseed_full.csv"
df_agg.to_csv(csv_agg,  index=False)
df_all.to_csv(csv_full, index=False)
print(f"\nSummary CSV  -> {csv_agg}")
print(f"Full CSV     -> {csv_full}")

# ── Identify best alpha ───────────────────────────────────────────────────────
calib_rows = df_agg[df_agg["alpha"] != "rolling"].copy()
calib_rows["alpha"] = calib_rows["alpha"].astype(float)

# Sweet spot: best mean score with skip rate in [15, 35] %
sweet = calib_rows[(calib_rows["mean_skip_%"] >= 15) &
                   (calib_rows["mean_skip_%"] <= 35)]
if not sweet.empty:
    best_row = sweet.loc[sweet["mean_score"].idxmax()]
    print(f"\n  Sweet spot (15-35% skip): a = {best_row['alpha']}  "
          f"mean={best_row['mean_score']:.3f}  skip={best_row['mean_skip_%']:.1f}%  "
          f"delta={best_row['delta_vs_rolling']:+.3f}")

best_overall = calib_rows.loc[calib_rows["mean_score"].idxmax()]
print(f"  Best overall:             a = {best_overall['alpha']}  "
      f"mean={best_overall['mean_score']:.3f}  skip={best_overall['mean_skip_%']:.1f}%  "
      f"delta={best_overall['delta_vs_rolling']:+.3f}")


# ── Plot 1: mean score ± std vs alpha + skip rate vs alpha ────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

alphas_f   = calib_rows["alpha"].tolist()
mean_scores = calib_rows["mean_score"].tolist()
std_scores  = calib_rows["std_score"].tolist()
mean_skips  = calib_rows["mean_skip_%"].tolist()
std_skips   = calib_rows["std_skip_%"].tolist()
deltas      = calib_rows["delta_vs_rolling"].tolist()

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    f"Alpha sweep — {len(args.seeds)} seeds each   {Path(args.csv).stem}   "
    f"N={M}  T={args.T}  floor={args.floor_frac}  time={args.time}s",
    fontsize=11, y=1.02,
)

# ── Left: mean score ± std vs alpha ──────────────────────────────────────────
ax = axes[0]
ax.errorbar(alphas_f, mean_scores, yerr=std_scores,
            fmt="o-", color="#E67E22", linewidth=2, markersize=7,
            capsize=5, capthick=1.5, elinewidth=1.5, zorder=3)
ax.axhline(rolling_mean, color="#27AE60", linewidth=1.8, linestyle="--",
           label=f"Rolling mean = {rolling_mean:.3f}")
ax.axhline(rolling_mean - roll_sub["best_score"].std(ddof=1),
           color="#27AE60", linewidth=0.8, linestyle=":", alpha=0.6)
ax.axhline(rolling_mean + roll_sub["best_score"].std(ddof=1),
           color="#27AE60", linewidth=0.8, linestyle=":", alpha=0.6,
           label=f"Rolling ± std")
# Shade sweet-spot skip range
ax.set_xscale("log")
if not sweet.empty:
    bx = best_row["alpha"]
    ax.axvline(bx, color="#C0392B", linewidth=1.5, linestyle=":",
               label=f"Sweet spot α={bx}")
    ax.scatter([bx], [best_row["mean_score"]], color="#C0392B", s=120, zorder=5)
ax.set_xlabel("α  (log scale)", fontsize=11)
ax.set_ylabel("Mean best UE score", fontsize=11)
ax.set_title(f"Mean score ± std across {len(args.seeds)} seeds", fontsize=10, pad=6)
ax.legend(fontsize=8.5, frameon=False)
ax.grid(axis="y", alpha=0.2, linestyle="--")

# ── Middle: delta vs rolling ──────────────────────────────────────────────────
ax2 = axes[1]
colors_d = ["#27AE60" if d > 0.02 else ("#F39C12" if abs(d) <= 0.02 else "#C0392B")
            for d in deltas]
ax2.bar(range(len(alphas_f)), deltas, color=colors_d,
        edgecolor="white", linewidth=0.8, zorder=2)
ax2.errorbar(range(len(alphas_f)), deltas, yerr=std_scores,
             fmt="none", color="black", capsize=4, elinewidth=1.2, zorder=3)
ax2.axhline(0, color="#555", linewidth=1.2)
ax2.set_xticks(range(len(alphas_f)))
ax2.set_xticklabels([str(a) for a in alphas_f], rotation=40, ha="right", fontsize=8.5)
ax2.set_xlabel("α", fontsize=11)
ax2.set_ylabel("Mean Δ vs rolling  (± std)", fontsize=11)
ax2.set_title("Calibration gain over rolling\n(green=better, red=worse)", fontsize=10, pad=6)
ax2.grid(axis="y", alpha=0.2, linestyle="--")
for xi, (d, s) in enumerate(zip(deltas, std_scores)):
    ax2.text(xi, d + s + 0.02, f"{d:+.2f}", ha="center", fontsize=8,
             fontweight="bold", color=colors_d[xi])

# ── Right: skip rate vs alpha ─────────────────────────────────────────────────
ax3 = axes[2]
ax3.errorbar(alphas_f, mean_skips, yerr=std_skips,
             fmt="s-", color="#8E44AD", linewidth=2, markersize=7,
             capsize=5, capthick=1.5, elinewidth=1.5)
ax3.axhline(roll_sub["skip_rate"].mean()*100, color="#27AE60", linewidth=1.5,
            linestyle="--", label=f"Rolling skip {roll_sub['skip_rate'].mean()*100:.1f}%")
ax3.axhspan(15, 35, alpha=0.08, color="#27AE60", label="Target zone 15–35%")
ax3.set_xscale("log")
ax3.set_xlabel("α  (log scale)", fontsize=11)
ax3.set_ylabel("Mean skip rate (%)", fontsize=11)
ax3.set_title("Skip rate vs α\n(shaded = target operating zone 15–35%)", fontsize=10, pad=6)
ax3.legend(fontsize=9, frameon=False)
ax3.grid(axis="y", alpha=0.2, linestyle="--")

plt.tight_layout()
plot1_path = OUT_DIR / "alpha_sweep_multiseed_main.png"
fig.savefig(plot1_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Plot -> {plot1_path}")


# ── Plot 2: per-seed scores for each alpha (strip + box) ─────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
fig2.suptitle(
    f"Per-seed scores by α  ({len(args.seeds)} seeds)  —  {Path(args.csv).stem}",
    fontsize=11, y=1.02,
)

# Left: individual score per seed per alpha
ax4 = axes2[0]
rng_jit = np.random.default_rng(0)
cmap    = plt.cm.plasma
for xi, alpha in enumerate(alphas_f):
    sub = df_all[(df_all["method"] == "calibration") & (df_all["alpha"] == alpha)]
    sc  = [r for _, r in sub["best_score"].items()]
    c   = cmap(0.1 + 0.8 * xi / max(len(alphas_f) - 1, 1))
    jit = rng_jit.uniform(-0.2, 0.2, len(sc))
    ax4.scatter([xi + j for j in jit], sc, color=c, s=60,
                zorder=3, edgecolors="white", linewidths=0.6)
    ax4.plot([xi - 0.3, xi + 0.3],
             [np.mean(sc), np.mean(sc)], color=c, linewidth=2.5)

# Rolling reference lines per seed
roll_scores_per_seed = {r["seed"]: r["best_score"]
                        for _, r in df_all[df_all["method"]=="rolling"].iterrows()}
for s_score in roll_scores_per_seed.values():
    ax4.axhline(s_score, color="#27AE60", linewidth=0.7, linestyle="--", alpha=0.4)
ax4.axhline(rolling_mean, color="#27AE60", linewidth=2.0, linestyle="--",
            label=f"Rolling mean = {rolling_mean:.3f}")
ax4.set_xticks(range(len(alphas_f)))
ax4.set_xticklabels([str(a) for a in alphas_f], rotation=40, ha="right", fontsize=8.5)
ax4.set_xlabel("α", fontsize=11)
ax4.set_ylabel("Best UE score", fontsize=11)
ax4.set_title("Individual seed scores per α\n(horizontal bar = mean)", fontsize=10, pad=6)
ax4.legend(fontsize=9, frameon=False)
ax4.grid(axis="y", alpha=0.2, linestyle="--")
ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

# Right: skip rate per alpha per seed
ax5 = axes2[1]
for xi, alpha in enumerate(alphas_f):
    sub = df_all[(df_all["method"] == "calibration") & (df_all["alpha"] == alpha)]
    sk  = [r * 100 for _, r in sub["skip_rate"].items()]
    c   = cmap(0.1 + 0.8 * xi / max(len(alphas_f) - 1, 1))
    jit = rng_jit.uniform(-0.2, 0.2, len(sk))
    ax5.scatter([xi + j for j in jit], sk, color=c, s=60,
                zorder=3, edgecolors="white", linewidths=0.6)
    ax5.plot([xi - 0.3, xi + 0.3], [np.mean(sk), np.mean(sk)], color=c, linewidth=2.5)

ax5.axhspan(15, 35, alpha=0.10, color="#27AE60", label="Target 15–35%")
ax5.axhline(roll_sub["skip_rate"].mean()*100, color="#27AE60",
            linewidth=2.0, linestyle="--", label=f"Rolling mean skip")
ax5.set_xticks(range(len(alphas_f)))
ax5.set_xticklabels([str(a) for a in alphas_f], rotation=40, ha="right", fontsize=8.5)
ax5.set_xlabel("α", fontsize=11)
ax5.set_ylabel("Skip rate (%)", fontsize=11)
ax5.set_title("Skip rate per α (shaded = target zone)", fontsize=10, pad=6)
ax5.legend(fontsize=9, frameon=False)
ax5.grid(axis="y", alpha=0.2, linestyle="--")
ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

plt.tight_layout()
plot2_path = OUT_DIR / "alpha_sweep_multiseed_perseed.png"
fig2.savefig(plot2_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig2)
plt.rcdefaults()
print(f"Plot -> {plot2_path}")

ue_eval.dispose()
print("\nDone.")
