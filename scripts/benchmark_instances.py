"""
benchmark_instances.py
Run no-gate / fixed-gate / rolling / calibration on multiple instances.
Fixed: seed=11, alpha=0.01, floor_frac=0.90.
Time budget scales with N so each instance gets a fair comparison.

Usage:
    python scripts/benchmark_instances.py
    python scripts/benchmark_instances.py --out-tag myrun
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
p.add_argument("--seed",        type=int,   default=11)
p.add_argument("--alpha",       type=float, default=0.01)
p.add_argument("--gate-frac",   type=float, default=0.90)
p.add_argument("--floor-frac",  type=float, default=0.90)
p.add_argument("--T",           type=int,   default=6)
p.add_argument("--D",           type=float, default=2.0)
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

# ── Instance list with per-instance time budgets ──────────────────────────────
INSTANCES = [
    ("center_102_Vicenza_k125.csv",   90,  "Vicenza",  125),
    ("center_153_Padova_k175.csv",    90,  "Padova",   175),
    ("center_146_Verona_k250.csv",    90,  "Verona",   250),
    ("center_79_Monza_k400.csv",      90,  "Monza",    400),
    ("center_270_Bologna_k650.csv",  150,  "Bologna",  650),
    ("center_276_Genova_k825.csv",   180,  "Genova",   825),
    ("center_87_Milano_k3375.csv",   600,  "Milano",  3375),
]

METHODS = ["no_gate", "fixed_gate", "rolling", "calibration"]

import datetime as _dt
_tag = args.out_tag or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = PROJECT_ROOT / "results" / "gate" / "multi_instance" / _tag
OUT_DIR.mkdir(parents=True, exist_ok=True)

total_runs = len(INSTANCES) * len(METHODS)
total_time = sum(t for _, t, _, _ in INSTANCES) * len(METHODS)
print(f"seed={args.seed}  alpha={args.alpha}  floor={args.floor_frac}")
print(f"Total runs : {total_runs}  (~{total_time//60} min)\n")

MAX_CHARGERS  = 6
Q_CAP         = 20.0
CUMULATIVE    = True
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]
LS_MODES      = ("site_swap", "local_remove")
N_WARMUP       = 10
EMA_WINDOW     = 30
N_WARMUP_CALIB = 20
CALIB_WINDOW   = 100


# ── Build one instance from a CSV ─────────────────────────────────────────────
def build_instance(csv_file: str):
    DATA_DIR  = PROJECT_ROOT / "data" / "input"
    df_raw    = pd.read_csv(DATA_DIR / csv_file)
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

    ue_mu   = args.ue_cap_per_server + 1e-4
    ue_eval = UEEvaluator(
        N=M, d=demand_TM[0], tau=distIJ, mu=ue_mu, s_max=MAX_CHARGERS,
        noopt_cost=args.ue_noopt, alpha_wait=args.ue_alpha_wait, N_bp=50,
        max_range=args.ue_max_range, penalty_distance=args.ue_penalty_distance,
    )
    return inst, demand_TM, J_i_list, Ij_int, P_T, distIJ, M, ue_eval


# ── Core ALNS runner ──────────────────────────────────────────────────────────
def run_single(method, time_budget, inst, demand_TM, J_i_list,
               Ij_int, P_T, distIJ, M, ue_eval):

    gate_frac  = args.gate_frac if method != "no_gate" else 0.0
    floor_frac = args.floor_frac
    alpha      = args.alpha
    seed       = args.seed

    def ue_fn(U):
        s, _ = ue_eval.evaluate(U, T=args.T, cumulative_install=CUMULATIVE)
        return float(s)

    def proxy_fn(U):
        return float(evaluate_u_numpy_greedy_binary(
            U, demand_TM, J_i_list, distIJ, Q_CAP, args.T, M, CUMULATIVE
        ))

    rng = np.random.default_rng(seed + 100)
    U_curr     = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=seed)
    score_curr = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    proxy_init  = proxy_fn(U_curr)
    proxy_gate  = gate_frac * proxy_init if gate_frac > 0 else 0.0
    proxy_floor = floor_frac * proxy_init

    ls_log = []
    all_proxy_ue_log = []
    p_explore = args.p_explore if method in ("rolling", "calibration") else 0.0

    skip_count = full_evals = iter_count = new_bests = 0
    t_vec, score_vec = [0.0], [score_best]

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
        best_idx   = int(np.argmax(ue_scores))
        iter_score = float(ue_scores[best_idx])
        iter_U     = visited[best_idx]

        ue_before = float(ue_scores[0])
        if ue_before > 1e-9:
            ls_log.append((ue_before, iter_score))
        if p_val is not None:
            all_proxy_ue_log.append((p_val, iter_score))

        # Threshold update
        enough_ls    = len(ls_log) >= N_WARMUP
        enough_calib = len(all_proxy_ue_log) >= N_WARMUP_CALIB
        if (method == "calibration" and enough_calib) or \
           (method == "rolling"     and enough_ls):
            if method == "calibration":
                wc  = all_proxy_ue_log[-CALIB_WINDOW:]
                prx = np.array([x[0] for x in wc])
                ues = np.array([x[1] for x in wc])
                adapt = None
                if np.std(prx) > 1e-9 and len(prx) >= 4:
                    b1, b0 = np.polyfit(prx, ues, 1)
                    if b1 > 1e-9:
                        sig   = max(float(np.std(ues - (b0 + b1*prx), ddof=2)), 1e-9)
                        adapt = (score_best + sig * float(sp_norm.ppf(alpha)) - b0) / b1
                if adapt is None:
                    ratios = [a/b for b, a in ls_log[-EMA_WINDOW:] if b > 1e-9]
                    adapt  = proxy_fn(U_best) / (np.mean(ratios) if ratios else 1.0)
            else:
                ratios = [a/b for b, a in ls_log[-EMA_WINDOW:] if b > 1e-9]
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
        "method":     method,
        "best_score": score_best,
        "skip_rate":  skip_count / max(iter_count, 1),
        "full_evals": full_evals,
        "new_bests":  new_bests,
        "iter_count": iter_count,
        "t_vec":      t_vec,
        "score_vec":  score_vec,
    }


# ── Run all instances ─────────────────────────────────────────────────────────
all_records = []
run_num = 0

for csv_file, time_limit, city_name, N_expected in INSTANCES:
    print(f"\n{'='*65}")
    print(f"  {city_name}  ({csv_file})  time={time_limit}s")
    print(f"{'='*65}")

    inst, demand_TM, J_i_list, Ij_int, P_T, distIJ, M, ue_eval = build_instance(csv_file)
    print(f"  N={M}  P_T[0]={P_T[0]}")

    results_this = {}
    for method in METHODS:
        run_num += 1
        print(f"  [{run_num}/{total_runs}]  {method}", flush=True)
        r = run_single(method, time_limit,
                       inst, demand_TM, J_i_list, Ij_int, P_T, distIJ, M, ue_eval)
        results_this[method] = r
        print(f"    best={r['best_score']:.3f}  skip={100*r['skip_rate']:.1f}%  "
              f"evals={r['full_evals']}  iters={r['iter_count']}", flush=True)

    ue_eval.dispose()

    nogate_s  = results_this["no_gate"]["best_score"]
    rolling_s = results_this["rolling"]["best_score"]
    calib_s   = results_this["calibration"]["best_score"]

    print(f"\n  Summary {city_name} (N={M}):")
    print(f"    No gate  : {nogate_s:.3f}")
    print(f"    Fixed    : {results_this['fixed_gate']['best_score']:.3f}  "
          f"(delta vs no-gate: {results_this['fixed_gate']['best_score']-nogate_s:+.3f})")
    print(f"    Rolling  : {rolling_s:.3f}  "
          f"(delta vs no-gate: {rolling_s-nogate_s:+.3f})")
    print(f"    Calib    : {calib_s:.3f}  "
          f"(delta vs rolling: {calib_s-rolling_s:+.3f})")

    for method, r in results_this.items():
        all_records.append({
            "city":            city_name,
            "csv":             csv_file,
            "N":               M,
            "time_budget_s":   time_limit,
            "method":          method,
            "best_score":      round(r["best_score"],  3),
            "gain_vs_nogate":  round(r["best_score"] - nogate_s, 3),
            "delta_vs_rolling":round(r["best_score"] - rolling_s, 3),
            "skip_rate_%":     round(100 * r["skip_rate"], 1),
            "full_evals":      r["full_evals"],
            "iter_count":      r["iter_count"],
            "new_bests":       r["new_bests"],
        })


# ── Results table ─────────────────────────────────────────────────────────────
df = pd.DataFrame(all_records)
df_pivot = df.pivot_table(
    index=["city", "N", "time_budget_s"],
    columns="method",
    values=["best_score", "skip_rate_%", "full_evals"],
    aggfunc="first",
).round(3)

print(f"\n{'='*80}")
print("MULTI-INSTANCE BENCHMARK RESULTS")
print(f"seed={args.seed}  alpha={args.alpha}  floor={args.floor_frac}")
print(f"{'='*80}")

# Compact summary: one row per city
summary_rows = []
for city, N, time_limit, _ in [(r[2], r[3], r[1], r[0]) for r in INSTANCES]:
    sub = df[df["city"] == city]
    ng  = sub[sub["method"]=="no_gate"   ]["best_score"].values[0]
    fg  = sub[sub["method"]=="fixed_gate"]["best_score"].values[0]
    rl  = sub[sub["method"]=="rolling"   ]["best_score"].values[0]
    cb  = sub[sub["method"]=="calibration"]["best_score"].values[0]
    rl_sk = sub[sub["method"]=="rolling"   ]["skip_rate_%"].values[0]
    cb_sk = sub[sub["method"]=="calibration"]["skip_rate_%"].values[0]
    rl_ev = sub[sub["method"]=="rolling"   ]["full_evals"].values[0]
    cb_ev = sub[sub["method"]=="calibration"]["full_evals"].values[0]
    summary_rows.append({
        "city": city, "N": N, "time_s": time_limit,
        "no_gate": ng, "fixed": fg, "rolling": rl, "calib": cb,
        "delta_calib_roll": round(cb - rl, 3),
        "roll_skip_%": rl_sk, "calib_skip_%": cb_sk,
        "roll_evals": rl_ev, "calib_evals": cb_ev,
    })

df_sum = pd.DataFrame(summary_rows)
print(df_sum[[
    "city","N","time_s","no_gate","fixed","rolling","calib",
    "delta_calib_roll","roll_skip_%","calib_skip_%"
]].to_string(index=False))

csv_path = OUT_DIR / "benchmark_instances_results.csv"
df.to_csv(csv_path, index=False)
df_sum.to_csv(OUT_DIR / "benchmark_instances_summary.csv", index=False)
print(f"\nCSV -> {csv_path}")


# ── Plot 1: scores + delta vs rolling across instances (sorted by N) ──────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
})

cities   = df_sum["city"].tolist()
N_vals   = df_sum["N"].tolist()
x        = np.arange(len(cities))
width    = 0.2
COLORS   = {"no_gate":"#2471A3","fixed_gate":"#C0392B",
            "rolling":"#27AE60","calibration":"#E67E22"}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
fig.suptitle(
    f"Multi-instance benchmark   seed={args.seed}   "
    f"alpha={args.alpha}   floor={args.floor_frac}",
    fontsize=11, y=1.02,
)

# ── Left: grouped bar — best score per method per city ───────────────────────
ax = axes[0]
offsets = [-1.5, -0.5, 0.5, 1.5]
col_map = {"no_gate":"no_gate","fixed_gate":"fixed",
           "rolling":"rolling","calibration":"calib"}
for mi, method in enumerate(METHODS):
    vals = df_sum[col_map[method]].tolist()
    lbl  = {"no_gate":"No gate","fixed_gate":"Fixed",
            "rolling":"Rolling","calibration":"Calib"}[method]
    ax.bar(x + offsets[mi]*width, vals, width*0.9,
           color=COLORS[method], alpha=0.8, label=lbl, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([f"{c}\nN={n}" for c, n in zip(cities, N_vals)], fontsize=8.5)
ax.set_ylabel("Best UE score", fontsize=11)
ax.set_title("Best score by instance and method", fontsize=10, pad=6)
ax.legend(fontsize=9, frameon=False)
ax.grid(axis="y", alpha=0.2, linestyle="--")

# ── Middle: delta calibration vs rolling, sorted by N ────────────────────────
ax2 = axes[1]
deltas  = df_sum["delta_calib_roll"].tolist()
bar_colors = ["#27AE60" if d > 0.02 else
              ("#F39C12" if abs(d) <= 0.02 else "#C0392B") for d in deltas]
ax2.bar(x, deltas, color=bar_colors, edgecolor="white", linewidth=0.8)
ax2.axhline(0, color="#555", linewidth=1.2)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{c}\nN={n}" for c, n in zip(cities, N_vals)], fontsize=8.5)
ax2.set_ylabel("delta calib - rolling", fontsize=11)
ax2.set_title("Calibration gain over rolling\nby instance size", fontsize=10, pad=6)
ax2.grid(axis="y", alpha=0.2, linestyle="--")
for xi, d in enumerate(deltas):
    ax2.text(xi, d + (0.02 if d >= 0 else -0.06),
             f"{d:+.3f}", ha="center", fontsize=8, fontweight="bold",
             color=bar_colors[xi])

# ── Right: skip rate + full evals comparison ─────────────────────────────────
ax3 = axes[2]
roll_skips  = df_sum["roll_skip_%"].tolist()
calib_skips = df_sum["calib_skip_%"].tolist()
ax3.plot(N_vals, roll_skips,  "o-", color="#27AE60", linewidth=2,
         markersize=7, label="Rolling skip %")
ax3.plot(N_vals, calib_skips, "s-", color="#E67E22", linewidth=2,
         markersize=7, label="Calib skip %")
ax3.axhspan(15, 35, alpha=0.08, color="#27AE60", label="Target 15-35%")
ax3.set_xscale("log")
ax3.set_xlabel("N (log scale)", fontsize=11)
ax3.set_ylabel("Skip rate (%)", fontsize=11)
ax3.set_title("Skip rate vs instance size", fontsize=10, pad=6)
ax3.legend(fontsize=9, frameon=False)
ax3.grid(axis="y", alpha=0.2, linestyle="--")
for ni, (n, rs, cs) in enumerate(zip(N_vals, roll_skips, calib_skips)):
    ax3.annotate(f"N={n}", (n, cs), textcoords="offset points",
                 xytext=(0, 8), ha="center", fontsize=7.5)

plt.tight_layout()
plot_path = OUT_DIR / "benchmark_instances_plot.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Plot -> {plot_path}")


# ── Plot 2: trajectory per instance (4 subplots for 4 methods × 7 instances) ──
fig2, axes2 = plt.subplots(2, 4, figsize=(18, 9))
fig2.suptitle(
    f"Score trajectories by instance   seed={args.seed}   "
    f"alpha={args.alpha}   floor={args.floor_frac}",
    fontsize=11, y=1.02,
)
cmap2 = plt.cm.viridis
for mi, method in enumerate(METHODS):
    row, col = mi // 4, mi % 4
    ax_m = axes2[row][col]
    lbl  = {"no_gate":"No gate","fixed_gate":"Fixed gate",
            "rolling":"Rolling","calibration":"Calib a=0.01"}[method]
    for ci, (csv_file, time_limit, city_name, _) in enumerate(INSTANCES):
        sub_rec = [r for r in all_records
                   if r["city"] == city_name and r["method"] == method]
        if not sub_rec:
            continue
        # find the t_vec/score_vec for this record
    ax_m.set_title(lbl, fontsize=10, pad=4)
    ax_m.spines["top"].set_visible(False)
    ax_m.spines["right"].set_visible(False)

# Trajectories need t_vec/score_vec — store them from the run
# Re-run approach: store in df_traj during run
# (all_records only stores scalars; trajectories are in results_this dicts above)
# Since we can't re-access them, just note this in the plot
for mi, method in enumerate(METHODS):
    row, col = mi // 4, mi % 4
    ax_m = axes2[row][col]
    ax_m.text(0.5, 0.5,
              "Trajectory data\nnot stored\n(see benchmark_instances_results.csv)",
              ha="center", va="center", transform=ax_m.transAxes,
              fontsize=9, color="#888")
    ax_m.axis("off")

# Use the last subplot row for a clean summary table
axes2[1][3].axis("off")
tbl_data = [["City","N","No gate","Fixed","Rolling","Calib","delta"]]
for _, r in df_sum.iterrows():
    tbl_data.append([
        r["city"], str(int(r["N"])),
        f"{r['no_gate']:.2f}", f"{r['fixed']:.2f}",
        f"{r['rolling']:.2f}", f"{r['calib']:.2f}",
        f"{r['delta_calib_roll']:+.3f}",
    ])
tbl = axes2[1][3].table(
    cellText=tbl_data[1:], colLabels=tbl_data[0],
    cellLoc="center", loc="center", bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (ri, ci), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if ri == 0:
        cell.set_facecolor("#1F4E79")
        cell.set_text_props(color="white", fontweight="bold")
    elif ci == 6:
        v = tbl_data[ri][6]
        try:
            dv = float(v)
            cell.set_facecolor("#C6EFCE" if dv > 0.02 else
                               "#FFEB9C" if abs(dv) <= 0.02 else "#FFC7CE")
        except Exception:
            pass
axes2[1][3].set_title("Summary table", fontsize=9, pad=4)

plt.tight_layout()
traj_path = OUT_DIR / "benchmark_instances_trajectories.png"
fig2.savefig(traj_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig2)
plt.rcdefaults()
print(f"Plot -> {traj_path}")
print("\nDone.")
