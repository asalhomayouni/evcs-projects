"""
benchmark_so_fix.py  —  Gate benchmark on a single instance with LS diagnostic.
Tracks whether new best solutions come from LS improving U_recon,
or from U_recon itself already being better than the incumbent (D&R contribution).

Usage:
    python scripts/benchmark_so_fix.py --csv center_58_Trieste_k225.csv --time 90
    python scripts/benchmark_so_fix.py --csv center_710_Palermo_k1025.csv --time 240
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

p = argparse.ArgumentParser()
p.add_argument("--csv",         required=True)
p.add_argument("--T",           type=int,   default=6)
p.add_argument("--D",           type=float, default=2.0)
p.add_argument("--seed",        type=int,   default=11)
p.add_argument("--time",        type=float, default=90.0)
p.add_argument("--alpha",       type=float, default=0.01)
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
args = p.parse_args()

city_name = Path(args.csv).stem.split("_")[2]
OUT_DIR   = PROJECT_ROOT / "results" / "gate" / "instance_tests" / city_name
OUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_CHARGERS  = 6
Q_CAP         = 20.0
CUMULATIVE    = True
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]
LS_MODES      = ("site_swap", "local_remove")
N_WARMUP       = 10
EMA_WINDOW     = 30
N_WARMUP_CALIB = 20
CALIB_WINDOW   = 100

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

print(f"Instance : {args.csv}   N={M}   T={args.T}   time={args.time}s")
print(f"alpha={args.alpha}   floor={args.floor_frac}   seed={args.seed}\n")

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


def run_single(method: str) -> dict:
    gate_frac  = args.gate_frac if method != "no_gate" else 0.0

    rng = np.random.default_rng(args.seed + 100)
    U_curr     = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=args.seed)
    score_curr = ue_fn(U_curr)
    U_best, score_best = dict(U_curr), score_curr

    proxy_init  = proxy_fn(U_curr)
    proxy_gate  = gate_frac * proxy_init if gate_frac > 0 else 0.0
    proxy_floor = args.floor_frac * proxy_init

    ls_log = []
    all_proxy_ue_log = []
    p_explore = args.p_explore if method in ("rolling", "calibration") else 0.0

    skip_count = full_evals = iter_count = new_bests = 0
    ls_calls = ls_helped = 0
    best_found_by_ls       = 0  # incumbent updated AFTER local search improved entry
    best_found_without_ls  = 0  # incumbent updated directly from reconstructed solution
    ls_improvements = []

    # ── Timing accumulators ────────────────────────────────────────────────────
    t_destroy     = 0.0   # (a) destroy only
    t_reconstruct = 0.0   # (b) greedy reconstruction only
    t_proxy       = 0.0   # (c) proxy_fn calls: gate check + threshold update
    t_ue          = 0.0   # (d) UE (Gurobi) evaluations — inside LS and exploration
    t_ls_pure     = 0.0   # (e) local_search overhead excluding UE calls within it

    _ue_acc = [0.0]    # mutable buffer: total UE time so far (auto-incremented by wrapper)

    def timed_ue_fn(U):
        """Drop-in replacement for ue_fn that accumulates wall time."""
        _t = time.perf_counter()
        s  = ue_fn(U)
        _ue_acc[0] += time.perf_counter() - _t
        return s

    t_vec, score_vec = [0.0], [score_best]
    t0 = time.perf_counter()

    while True:
        if time.perf_counter() - t0 >= args.time:
            break
        iter_count += 1
        base = U_curr if rng.random() < 0.7 else U_best
        mode = str(rng.choice(DESTROY_MODES))

        # ── (a) Destroy ────────────────────────────────────────────────────────
        _t = time.perf_counter()
        U_partial, _ = destroy_multi_u(
            base, inst, rng, P_T, args.frac_remove, mode,
            site_cap=MAX_CHARGERS, cumulative_install=CUMULATIVE,
        )
        t_destroy += time.perf_counter() - _t

        # ── (b) Reconstruct ────────────────────────────────────────────────────
        _t = time.perf_counter()
        U_recon, _ = reconstruct_u_dict_fast(
            U_partial, demand_TM, P_T, Ij_int,
            U_cap=MAX_CHARGERS, Q=Q_CAP, rng=rng, cumulative_install=CUMULATIVE,
        )
        t_reconstruct += time.perf_counter() - _t

        # ── (b) Proxy gate evaluation ──────────────────────────────────────────
        gate_rejected = False
        if proxy_gate > 0:
            _t = time.perf_counter()
            p_val = proxy_fn(U_recon)
            t_proxy += time.perf_counter() - _t

            if p_val < proxy_gate:
                gate_rejected = True
                if rng.random() < p_explore:
                    # exploration: still run UE so gate can learn
                    ue_before = _ue_acc[0]
                    ue_s = timed_ue_fn(U_recon)
                    t_ue += _ue_acc[0] - ue_before
                    all_proxy_ue_log.append((p_val, ue_s))
                    full_evals += 1
                else:
                    skip_count += 1
        else:
            p_val = None

        p_explore = max(args.p_min, p_explore * args.p_decay)
        if gate_rejected:
            continue

        # ── (c/d) Local Search (UE inside tracked separately) ─────────────────
        ue_before_ls = _ue_acc[0]
        _t = time.perf_counter()
        visited, ue_scores, _ = local_search_u_proxy(
            U_recon, inst, rng, P_T,
            demand_TM, J_i_list, distIJ,
            Q_CAP, args.T, M, CUMULATIVE, MAX_CHARGERS,
            args.ls_moves, 0.08, LS_MODES,
            Ij_int=Ij_int, collect_visited=True, proxy_fn=timed_ue_fn,
        )
        t_ls_total    = time.perf_counter() - _t
        ue_in_ls      = _ue_acc[0] - ue_before_ls
        t_ue      += ue_in_ls
        t_ls_pure += t_ls_total - ue_in_ls

        full_evals += len(visited)
        best_idx    = int(np.argmax(ue_scores))
        iter_score  = float(ue_scores[best_idx])
        iter_U      = visited[best_idx]
        entry_score = float(ue_scores[0])

        # ── LS diagnostic ──────────────────────────────────────────────────────
        ls_calls += 1
        ls_imp = iter_score - entry_score
        ls_improvements.append(ls_imp)
        if ls_imp > 1e-6:
            ls_helped += 1
        if iter_score > score_best:
            new_bests += 1
            if ls_imp > 1e-6:
                best_found_by_ls += 1
            else:
                best_found_without_ls += 1

        if entry_score > 1e-9:
            ls_log.append((entry_score, iter_score))
        if p_val is not None:
            all_proxy_ue_log.append((p_val, iter_score))

        # ── Threshold update (proxy_fn(U_best) timed separately) ──────────────
        if (method == "calibration" and len(all_proxy_ue_log) >= N_WARMUP_CALIB) or \
           (method == "rolling"     and len(ls_log) >= N_WARMUP):
            if method == "calibration":
                wc  = all_proxy_ue_log[-CALIB_WINDOW:]
                prx = np.array([x[0] for x in wc])
                ues = np.array([x[1] for x in wc])
                adapt = None
                if np.std(prx) > 1e-9 and len(prx) >= 4:
                    b1, b0 = np.polyfit(prx, ues, 1)
                    if b1 > 1e-9:
                        sig   = max(float(np.std(ues-(b0+b1*prx), ddof=2)), 1e-9)
                        adapt = (score_best + sig * float(sp_norm.ppf(args.alpha)) - b0) / b1
                if adapt is None:
                    ratios = [a/b for b, a in ls_log[-EMA_WINDOW:] if b > 1e-9]
                    _t = time.perf_counter()
                    pval_best = proxy_fn(U_best)
                    t_proxy += time.perf_counter() - _t
                    adapt = pval_best / (np.mean(ratios) if ratios else 1.0)
            else:
                ratios = [a/b for b, a in ls_log[-EMA_WINDOW:] if b > 1e-9]
                _t = time.perf_counter()
                pval_best = proxy_fn(U_best)
                t_proxy += time.perf_counter() - _t
                adapt = pval_best / (np.mean(ratios) if ratios else 1.0)
            proxy_gate = min(adapt, proxy_floor)

        if iter_score >= score_curr:
            U_curr, score_curr = dict(iter_U), iter_score
        if iter_score > score_best:
            U_best, score_best = dict(iter_U), iter_score

        t_vec.append(time.perf_counter() - t0)
        score_vec.append(score_best)

    t_elapsed   = time.perf_counter() - t0
    t_dr        = t_destroy + t_reconstruct
    t_accounted = t_dr + t_proxy + t_ue + t_ls_pure
    t_other     = max(0.0, t_elapsed - t_accounted)

    return {
        "method":            method,
        "best_score":        score_best,
        "skip_%":            round(100 * skip_count / max(iter_count, 1), 1),
        "full_evals":        full_evals,
        "iter_count":        iter_count,
        "new_bests":         new_bests,
        "ls_calls":          ls_calls,
        "ls_helped_%":            round(100 * ls_helped / max(ls_calls, 1), 1),
        "mean_ls_improv":         round(float(np.mean(ls_improvements)) if ls_improvements else 0.0, 5),
        "best_found_by_ls":       best_found_by_ls,
        "best_found_without_ls":  best_found_without_ls,
        "ls_responsible_%":       round(100 * best_found_by_ls / max(new_bests, 1), 1),
        "dr_responsible_%":       round(100 * best_found_without_ls / max(new_bests, 1), 1),
        # timing — split and combined
        "budget_s":            round(t_elapsed, 2),
        "t_destroy_s":         round(t_destroy, 2),
        "t_reconstruct_s":     round(t_reconstruct, 2),
        "t_dr_s":              round(t_dr, 2),
        "t_proxy_s":           round(t_proxy, 2),
        "t_ue_s":              round(t_ue, 2),
        "t_ls_pure_s":         round(t_ls_pure, 2),
        "t_other_s":           round(t_other, 2),
        "pct_destroy":         round(100 * t_destroy     / max(t_elapsed, 1e-9), 1),
        "pct_reconstruct":     round(100 * t_reconstruct / max(t_elapsed, 1e-9), 1),
        "pct_dr":              round(100 * t_dr          / max(t_elapsed, 1e-9), 1),
        "pct_proxy":           round(100 * t_proxy       / max(t_elapsed, 1e-9), 1),
        "pct_ue":              round(100 * t_ue          / max(t_elapsed, 1e-9), 1),
        "pct_ls_pure":         round(100 * t_ls_pure     / max(t_elapsed, 1e-9), 1),
        "pct_other":           round(100 * t_other       / max(t_elapsed, 1e-9), 1),
        "avg_destroy_ms":      round(1000 * t_destroy     / max(iter_count, 1), 2),
        "avg_reconstruct_ms":  round(1000 * t_reconstruct / max(iter_count, 1), 2),
        "avg_dr_ms":           round(1000 * t_dr          / max(iter_count, 1), 2),
        "avg_proxy_ms":        round(1000 * t_proxy       / max(iter_count, 1), 2),
        "avg_ue_ms":           round(1000 * t_ue          / max(ls_calls,   1), 2),
        "avg_ls_pure_ms":      round(1000 * t_ls_pure     / max(ls_calls,   1), 2),
        "t_vec":               t_vec,
        "score_vec":           score_vec,
    }


METHODS = ["no_gate", "fixed_gate", "rolling", "calibration"]
results = {}
for method in METHODS:
    print(f"  [{method}]", flush=True)
    r = run_single(method)
    results[method] = r
    print(f"    best={r['best_score']:.3f}  skip={r['skip_%']}%  "
          f"evals={r['full_evals']}  iters={r['iter_count']}")
    print(f"    LS helped {r['ls_helped_%']}% of calls  "
          f"mean_improv={r['mean_ls_improv']:.5f}")
    print(f"    New bests={r['new_bests']}  "
          f"best_found_by_ls={r['best_found_by_ls']}  "
          f"best_found_without_ls={r['best_found_without_ls']}")
    print(f"    LS responsible for {r['ls_responsible_%']}% of incumbent updates  "
          f"(D&R responsible for {r['dr_responsible_%']}%)")
    print(f"    --- Time breakdown (budget={r['budget_s']:.1f}s) ---")
    print(f"    (a) Destroy:       {r['t_destroy_s']:6.2f}s  ({r['pct_destroy']:5.1f}%)  "
          f"avg/iter  {r['avg_destroy_ms']:7.1f} ms")
    print(f"    (b) Reconstruct:   {r['t_reconstruct_s']:6.2f}s  ({r['pct_reconstruct']:5.1f}%)  "
          f"avg/iter  {r['avg_reconstruct_ms']:7.1f} ms")
    print(f"    (a+b) D&R total:   {r['t_dr_s']:6.2f}s  ({r['pct_dr']:5.1f}%)")
    print(f"    (c) Proxy eval:    {r['t_proxy_s']:6.2f}s  ({r['pct_proxy']:5.1f}%)  "
          f"avg/iter  {r['avg_proxy_ms']:7.1f} ms")
    print(f"    (d) UE (Gurobi):   {r['t_ue_s']:6.2f}s  ({r['pct_ue']:5.1f}%)  "
          f"avg/LS    {r['avg_ue_ms']:7.1f} ms")
    print(f"    (e) LS overhead:   {r['t_ls_pure_s']:6.2f}s  ({r['pct_ls_pure']:5.1f}%)  "
          f"avg/LS    {r['avg_ls_pure_ms']:7.1f} ms")
    print(f"        Other/setup:   {r['t_other_s']:6.2f}s  ({r['pct_other']:5.1f}%)")

ue_eval.dispose()

nogate_s  = results["no_gate"]["best_score"]
rolling_s = results["rolling"]["best_score"]

rows = []
for method in METHODS:
    r = results[method]
    rows.append({
        "method":            method,
        "best_score":        round(r["best_score"], 3),
        "gain_vs_nogate":    round(r["best_score"] - nogate_s, 3),
        "delta_vs_rolling":  round(r["best_score"] - rolling_s, 3),
        "skip_%":            r["skip_%"],
        "full_evals":        r["full_evals"],
        "ls_helped_%":           r["ls_helped_%"],
        "mean_ls_improv":        r["mean_ls_improv"],
        "best_found_by_ls":      r["best_found_by_ls"],
        "best_found_without_ls": r["best_found_without_ls"],
        "ls_responsible_%":      r["ls_responsible_%"],
        "dr_responsible_%":      r["dr_responsible_%"],
        "budget_s":              r["budget_s"],
        "t_destroy_s":           r["t_destroy_s"],
        "t_reconstruct_s":       r["t_reconstruct_s"],
        "t_dr_s":                r["t_dr_s"],
        "t_proxy_s":             r["t_proxy_s"],
        "t_ue_s":                r["t_ue_s"],
        "t_ls_pure_s":           r["t_ls_pure_s"],
        "t_other_s":             r["t_other_s"],
        "pct_destroy":           r["pct_destroy"],
        "pct_reconstruct":       r["pct_reconstruct"],
        "pct_dr":                r["pct_dr"],
        "pct_proxy":             r["pct_proxy"],
        "pct_ue":                r["pct_ue"],
        "pct_ls_pure":           r["pct_ls_pure"],
        "pct_other":             r["pct_other"],
        "avg_destroy_ms":        r["avg_destroy_ms"],
        "avg_reconstruct_ms":    r["avg_reconstruct_ms"],
        "avg_dr_ms":             r["avg_dr_ms"],
        "avg_proxy_ms":          r["avg_proxy_ms"],
        "avg_ue_ms":             r["avg_ue_ms"],
        "avg_ls_pure_ms":        r["avg_ls_pure_ms"],
    })

df = pd.DataFrame(rows)
print(f"\n{'='*80}")
print(f"  {city_name}   N={M}   time={args.time}s   seed={args.seed}   alpha={args.alpha}")
print(f"{'='*80}")
print(df.to_string(index=False))
df.to_csv(OUT_DIR / f"{city_name}_results.csv", index=False)
print(f"\nCSV -> {OUT_DIR / f'{city_name}_results.csv'}")

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":"serif","font.size":10,
    "axes.spines.top":False,"axes.spines.right":False,
})
COLORS = {"no_gate":"#2471A3","fixed_gate":"#C0392B",
          "rolling":"#27AE60","calibration":"#E67E22"}
LABELS = {"no_gate":"No gate","fixed_gate":"Fixed",
          "rolling":"Rolling",
          "calibration":f"Calib\na={args.alpha}"}

fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 5))
fig.suptitle(
    f"{city_name}  N={M}  T={args.T}  time={args.time}s  "
    f"seed={args.seed}  alpha={args.alpha}  floor={args.floor_frac}",
    fontsize=11, y=1.02,
)

for method in METHODS:
    r = results[method]
    ax1.plot(r["t_vec"], r["score_vec"], color=COLORS[method], linewidth=2.0,
             label=f"{LABELS[method].replace(chr(10),' ')}  {r['best_score']:.2f}")
ax1.set_xlabel("Time (s)", fontsize=11)
ax1.set_ylabel("Best UE score", fontsize=11)
ax1.set_title("Score trajectory", fontsize=10, pad=6)
ax1.legend(fontsize=8.5, frameon=False)
ax1.grid(axis="y", alpha=0.2, linestyle="--")

ls_help = [results[m]["ls_helped_%"] for m in METHODS]
bars2 = ax2.bar(range(len(METHODS)), ls_help,
                color=[COLORS[m] for m in METHODS], alpha=0.8, edgecolor="white")
ax2.set_xticks(range(len(METHODS)))
ax2.set_xticklabels([LABELS[m] for m in METHODS], fontsize=8.5)
ax2.set_ylabel("% LS calls where LS improved entry", fontsize=10)
ax2.set_title("LS contribution rate\n(how often LS beats D&R entry score)", fontsize=10, pad=6)
ax2.grid(axis="y", alpha=0.2, linestyle="--")
for bar, v in zip(bars2, ls_help):
    ax2.text(bar.get_x()+bar.get_width()/2, v+0.5, f"{v:.1f}%",
             ha="center", fontsize=9, fontweight="bold")

xi    = np.arange(len(METHODS))
nb_ls = [results[m]["ls_responsible_%"] for m in METHODS]
nb_dr = [results[m]["dr_responsible_%"] for m in METHODS]
ax3.bar(xi-0.2, nb_ls, 0.38, color="#8E44AD", alpha=0.8,
        label="best_found_by_ls (LS improved entry)", edgecolor="white")
ax3.bar(xi+0.2, nb_dr, 0.38, color="#F39C12", alpha=0.8,
        label="best_found_without_ls (D&R direct)", edgecolor="white")
ax3.set_xticks(xi)
ax3.set_xticklabels([LABELS[m] for m in METHODS], fontsize=8.5)
ax3.set_ylabel("% of incumbent updates", fontsize=10)
ax3.set_title("LS responsible for X% of incumbent updates", fontsize=10, pad=6)
ax3.legend(fontsize=8.5, frameon=False)
ax3.grid(axis="y", alpha=0.2, linestyle="--")

# Panel 4 — stacked time breakdown (destroy and reconstruct split)
TIME_CATS   = ["(a) Destroy", "(b) Reconstruct", "(c) Proxy", "(d) UE", "(e) LS", "Other"]
TIME_COLORS = ["#1A5276", "#2471A3", "#F39C12", "#C0392B", "#8E44AD", "#AAB7B8"]
pct_keys    = ["pct_destroy", "pct_reconstruct", "pct_proxy", "pct_ue", "pct_ls_pure", "pct_other"]

xi    = np.arange(len(METHODS))
bottoms = np.zeros(len(METHODS))
for cat, col, key in zip(TIME_CATS, TIME_COLORS, pct_keys):
    vals = np.array([results[m][key] for m in METHODS], dtype=float)
    bars = ax4.bar(xi, vals, bottom=bottoms, color=col, alpha=0.85,
                   label=cat, edgecolor="white", linewidth=0.6)
    for bi, (v, b) in enumerate(zip(vals, bottoms)):
        if v >= 4:
            ax4.text(xi[bi], b + v / 2, f"{v:.0f}%",
                     ha="center", va="center", fontsize=7.5,
                     fontweight="bold", color="white")
    bottoms += vals

ax4.set_xticks(xi)
ax4.set_xticklabels([LABELS[m].replace("\n", " ") for m in METHODS], fontsize=8.5)
ax4.set_ylabel("% of time budget", fontsize=10)
ax4.set_title("Iteration time breakdown\n(a) D&R  (b) Proxy  (c) UE  (d) LS overhead", fontsize=9, pad=6)
ax4.set_ylim(0, 105)
ax4.legend(fontsize=7.5, frameon=False, loc="upper right")
ax4.grid(axis="y", alpha=0.2, linestyle="--")

plt.tight_layout()
plot_path = OUT_DIR / f"{city_name}_results.png"
fig.savefig(plot_path, dpi=160, bbox_inches="tight", facecolor="white")
plt.close(fig)
plt.rcdefaults()
print(f"Plot -> {plot_path}")
print("Done.")
