"""
Trajectory data-gathering experiment -- learning-based gate study.

Runs the ALNS loop with the proxy gate DISABLED (dumb mode / let everything
through) and records, for each destroy-repair candidate that enters local
search:

    proxy_0, proxy_1, ..., proxy_{ls_moves}  -- proxy score at every LS step
    full_eval                                  -- Gurobi LP score of U_best
                                                 (the proxy-best solution
                                                  found during that LS run)

After collecting ~N_TRAJ trajectories the script:
  1. Saves raw data -> results/trajectories/trajectories.csv
  2. Plots trajectory bundles coloured by final full-eval quality
  3. Plots per-step Pearson correlation with final full-eval
  4. Runs Linear Regression + Random-Forest regression and reports R²

Usage:
    python scripts/collect_trajectories.py
    python scripts/collect_trajectories.py --n-trajectories 500 --seed 7
    python scripts/collect_trajectories.py --no-plot          # headless
"""

# =========================
# IMPORTS
# =========================
from pathlib import Path
import argparse
import time
import sys

import numpy as np
import pandas as pd
import matplotlib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from evcs.geom import build_arcs
from evcs.proxy import evaluate_u_numpy_greedy_binary
from evcs.destroy import destroy_multi_u
from evcs.greedy import reconstruct_u_dict_fast
from evcs.methods import greedy_schedule_multi_from_variants
from evcs.local_search import local_search_u_proxy
from evcs.full_eval_grb import GRBEvaluator
from evcs.ue_evaluator import UEEvaluator

# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser(description="Collect proxy-trajectory data for learning-based gate study")
# --- instance parameters (overridden by --config-row if supplied) ---
parser.add_argument("--csv",            type=str,   default="center_146_Verona_k250.csv")
parser.add_argument("--seed",           type=int,   default=11)
parser.add_argument("--D",              type=float, default=2.0)
parser.add_argument("--T",              type=int,   default=6)
parser.add_argument("--Q",              type=float, default=20.0)
# --- configs.tsv job-array entry point ---
parser.add_argument("--config-row",     type=int,   default=None,
                    help="Row index in configs.tsv (0-based).  When set, "
                         "csv/seed/D/T/Q are read from that row and the "
                         "individual flags above are ignored.")
# --- trajectory collection ---
parser.add_argument("--n-trajectories", type=int,   default=1000,
                    help="Number of trajectories to collect (default 1000)")
parser.add_argument("--ls-moves",       type=int,   default=12,
                    help="Local search moves per trajectory (default 12)")
parser.add_argument("--no-plot",        action="store_true",
                    help="Skip matplotlib figures (headless / batch mode)")
# --- full-eval oracle ---
parser.add_argument("--full-eval",      type=str,   default="grb",
                    choices=["grb", "ue"],
                    help="Full evaluator: grb (default) or ue (UEEvaluator)")
parser.add_argument("--ue-mu",          type=float, default=7.5,
                    help="UE service rate per server (default 7.5)")
parser.add_argument("--ue-alpha-wait",  type=float, default=10.0,
                    help="UE waiting-cost weight alpha_wait (default 10.0)")
parser.add_argument("--ue-noopt",       type=float, default=30.0,
                    help="UE no-option LP penalty in km (default 30.0 — high enough that "
                         "in-range demand always prefers a station)")
parser.add_argument("--ue-max-range",   type=float, default=2.0,
                    help="UE hard distance cutoff: demand can only be assigned to stations "
                         "within this radius in km (default 2.0, matches D from build_arcs)")
parser.add_argument("--ue-penalty-distance", type=float, default=5.0,
                    help="UE scale factor on tau in objective (default 5.0 — amplifies "
                         "distance sensitivity within the 2 km range window)")
parser.add_argument("--ue-cap-per-server", type=float, default=4.0,
                    help="UE capacity per server for proxy (default 4.0, matches Julia station_cap)")
args = parser.parse_args()

# Set backend BEFORE importing pyplot -- must happen here, at module level
if args.no_plot:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# -----------------------------------------------------------------------
# If --config-row was given, load instance params from configs.tsv
# configs.tsv columns (tab-separated, no header):
#   row_id  csv_file  seed  D  T  Q
# -----------------------------------------------------------------------
CONFIGS_TSV = PROJECT_ROOT / "configs.tsv"

if args.config_row is not None:
    cfg = pd.read_csv(CONFIGS_TSV, sep="\t", header=None,
                      names=["row_id", "csv", "seed", "D", "T", "Q"])
    row = cfg[cfg["row_id"] == args.config_row].iloc[0]
    args.csv  = str(row["csv"]).strip()
    args.seed = int(row["seed"])
    args.D    = float(row["D"])
    args.T    = int(row["T"])
    args.Q    = float(row["Q"])
    print(f"[config-row {args.config_row}] {args.csv}  seed={args.seed}  D={args.D}  T={args.T}  Q={args.Q}")

DATA_DIR    = PROJECT_ROOT / "data" / "input"
RESULTS_DIR = PROJECT_ROOT / "results" / "trajectories"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# INSTANCE BUILD  (same logic as run_evcs.py)
# =========================
def build_instance_from_csv(csv_path, T, seed, D_km):
    df = pd.read_csv(csv_path)

    coords_deg = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(dtype=float)
    coords_km  = coords_deg.copy()
    coords_km[:, 0] *= 111.0
    coords_km[:, 1] *= 111.0

    pop = df["Aggregated_Population"].to_numpy(dtype=float)
    pop = np.maximum(pop, 0.0)
    if pop.sum() <= 0:
        raise ValueError("Population column sums to zero.")
    pop_share = pop / pop.sum()

    M    = coords_km.shape[0]
    base = pop_share * M   # sum = M

    rng_inst = np.random.default_rng(seed)
    demand_IT = np.zeros((T, M), dtype=float)
    for t in range(T):
        season_factor = 1.0 + 0.2 * np.sin(2 * np.pi * t / T)
        noise = rng_inst.normal(0.0, 0.05, size=M)
        demand_IT[t] = np.maximum(0.0, base * season_factor * (1.0 + noise))

    distIJ, in_range, Ji, Ij = build_arcs(
        coords_km, coords_km, D=D_km, forbid_self=False
    )

    return {
        "coords_I":  coords_km,
        "coords_J":  coords_km.copy(),
        "demand_IT": demand_IT,
        "distIJ":    distIJ,
        "in_range":  in_range,
        "Ji":        Ji,
        "Ij":        Ij,
        "M":         M,
        "N":         M,
    }


# =========================
# MAIN
# =========================
def main():
    N_TRAJ       = args.n_trajectories
    LS_MOVES     = args.ls_moves
    SEED         = args.seed
    T            = args.T
    Q            = args.Q
    D            = args.D
    SHOW_PLOT    = not args.no_plot

    # Algorithm hyper-params (match run_evcs.py defaults)
    P_T                   = [8] * T
    max_chargers_per_site = 6
    frac_remove           = 0.3
    ls_frac_remove        = 0.08
    ls_modes              = ("site_swap", "local_remove")
    destroy_modes         = ["site_swap", "local_remove", "area_destroy"]
    top_k_choice          = 3
    cumulative_install    = True
    policy                = "closest_priority"

    # Output sub-directory: one folder per instance+eval so parallel runs don't collide
    instance_tag = Path(args.csv).stem
    if args.config_row is not None:
        instance_tag = f"row{args.config_row}_{instance_tag}"
    instance_tag = f"{instance_tag}_{args.full_eval}"
    out_dir = RESULTS_DIR / instance_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load instance
    # -------------------------
    csv_path = DATA_DIR / args.csv
    print(f"Loading instance: {csv_path}")
    inst = build_instance_from_csv(csv_path, T=T, seed=SEED, D_km=D)

    M        = inst["M"]
    N        = inst["N"]
    distIJ   = inst["distIJ"]
    in_range = inst["in_range"]
    Ji       = inst["Ji"]
    Ij       = inst["Ij"]
    demand_IT = inst["demand_IT"]   # (T, M)

    demand_TM = np.asarray(demand_IT, dtype=float)
    if demand_TM.shape != (T, M):
        demand_TM = demand_TM.T

    # Arc helpers
    Ij_int = {j: [] for j in range(N)}
    Ji_int = {i: [] for i in range(M)}
    for (i, j) in in_range:
        Ij_int[j].append(i)
        Ji_int[i].append(j)

    J_i_list = [
        sorted(Ji_int[i], key=lambda j: float(distIJ[i, j]))
        for i in range(M)
    ]

    Q_cap = float(Q)

    print(f"  N={N}  T={T}  D={D}  |A|={len(in_range)}")

    # -------------------------
    # Full evaluator (GRB or UE)
    # -------------------------
    use_ue = (args.full_eval == "ue")
    if use_ue:
        ue_noopt   = args.ue_noopt           # no-option cost (km); default 30 = effectively infinite
        ue_cap     = args.ue_cap_per_server  # capacity per server for proxy; default 4.0 (Julia default)
        ue_mu      = ue_cap + 1e-4           # Erlang-C service rate = cap + eps (Julia convention)

        d_ue = demand_TM[0]
        ue_max_range      = args.ue_max_range
        ue_penalty_dist   = args.ue_penalty_distance
        print(
            f"Initialising UEEvaluator  "
            f"(mu={ue_mu:.4f}, noopt_cost={ue_noopt} km, alpha_wait={args.ue_alpha_wait}, "
            f"max_range={ue_max_range} km, penalty_distance={ue_penalty_dist}, "
            f"cap/server={ue_cap})"
        )
        ue_eval = UEEvaluator(
            N=N, d=d_ue, tau=distIJ, mu=ue_mu, s_max=max_chargers_per_site,
            noopt_cost=ue_noopt, alpha_wait=args.ue_alpha_wait, N_bp=50,
            max_range=ue_max_range, penalty_distance=ue_penalty_dist,
        )
        def full_eval_fn(U):
            score, _ = ue_eval.evaluate(U, T=T, cumulative_install=cumulative_install)
            return score, None
        def dispose_eval():
            ue_eval.dispose()

        # Binary greedy proxy (T=6, Q_cap=20) is the best available LS guide for UE full-eval.
        # Scope-matching to T=1/Q=ue_cap saturates at the capacity ceiling (r drops to 0.37).
        # T=6/Q=20 captures placement signal across all periods → r≈0.74 with UE.
        print(f"UE full-eval proxy (LS guide): binary greedy T={T}, Q_cap={Q_cap}")
        proxy_fn_override = None   # None → local_search_u_proxy uses binary greedy (default)
    else:
        print("Initialising GRBEvaluator ...")
        grb_eval = GRBEvaluator(M, N, T, in_range, demand_TM, Q_cap, cumulative_install)
        def full_eval_fn(U):
            return grb_eval.evaluate(U)
        def dispose_eval():
            grb_eval.dispose()
        proxy_fn_override = None   # use default GRB-based proxy inside LS

    # -------------------------
    # Initial solution
    # -------------------------
    rng = np.random.default_rng(SEED)

    U_curr = greedy_schedule_multi_from_variants(inst, P_T, D, seed=SEED)
    U_best = dict(U_curr)

    if proxy_fn_override is not None:
        proxy_curr = float(proxy_fn_override(U_curr))
    else:
        proxy_curr = float(evaluate_u_numpy_greedy_binary(
            U_curr, demand_TM, J_i_list, distIJ, Q_cap, T, N, cumulative_install
        ))

    best_full_score, _ = full_eval_fn(U_curr)

    print(f"  Initial proxy={proxy_curr:.4f}  full={best_full_score:.4f}")
    print(f"\nCollecting {N_TRAJ} trajectories (gate DISABLED) …\n")

    # -------------------------
    # Data collection loop
    # -------------------------
    records = []
    it = 0
    t0 = time.perf_counter()

    seen = set()
    seen_max = 2000

    def hash_u(U):
        return tuple(sorted(U.items()))

    seen.add(hash_u(U_curr))

    U_cap_per_site = max_chargers_per_site

    while len(records) < N_TRAJ:
        it += 1

        # 1) Destroy -> Reconstruct
        base = U_curr if rng.random() < 0.7 else U_best
        mode = str(rng.choice(destroy_modes))

        U_partial, _ = destroy_multi_u(
            base, inst, rng, P_T, frac_remove, mode,
            site_cap=U_cap_per_site, cumulative_install=cumulative_install
        )
        U_recon, _ = reconstruct_u_dict_fast(
            U_partial, demand_TM, P_T, Ij_int,
            U_cap=U_cap_per_site, Q=Q_cap, rng=rng,
            cumulative_install=cumulative_install,
        )

        # Skip exact duplicates
        h = hash_u(U_recon)
        if len(seen) < seen_max and h in seen:
            continue
        if len(seen) < seen_max:
            seen.add(h)

        # 2) Snapshot incumbent *before* LS so we can compute full_eval delta later
        incumbent_full_score = best_full_score

        # 3) Local search -- collect proxy trajectory
        #    Gate is DISABLED: every candidate goes through LS
        visited, proxy_scores, proxy_ls = local_search_u_proxy(
            U_recon, inst, rng, P_T,
            demand_TM, J_i_list, distIJ,
            Q_cap, T, N,
            cumulative_install,
            max_chargers_per_site,
            LS_MOVES, ls_frac_remove, ls_modes,
            Ij_int=Ij_int,
            top_k_choice=top_k_choice,
            collect_visited=True,
            proxy_fn=proxy_fn_override,
        )
        # proxy_scores: list of length LS_MOVES+1
        #   proxy_scores[0]  = proxy(U_recon) before any LS move
        #   proxy_scores[k]  = proxy of candidate generated at LS step k

        # 4) Full eval of U_best found by LS (the proxy-maximising solution)
        #    U_best after LS = the visited solution with the highest running proxy.
        #    We find it by tracking argmax of the running max.
        running_best_proxy = proxy_scores[0]
        U_ls_best = visited[0]
        for k, (U_v, p) in enumerate(zip(visited[1:], proxy_scores[1:]), start=1):
            if p > running_best_proxy + 1e-9:
                running_best_proxy = p
                U_ls_best = U_v

        full_score, _ = full_eval_fn(U_ls_best)

        # 5) Update incumbent (optional -- keeps the ALNS "alive")
        if full_score >= best_full_score - 1e-4:
            U_curr = dict(U_ls_best)
        if full_score > best_full_score:
            best_full_score = full_score
            U_best = dict(U_ls_best)

        # 6) Record trajectory
        #    incumbent_full_eval = best_full_score *before* this LS run
        #    full_eval_delta     = improvement over that incumbent (negative = worse)
        #    proxy_scores may be shorter than ls_moves+1 if LS stopped early;
        #    pad with None so every row has the same columns (-> NaN in CSV/DataFrame)
        record = {
            "traj_id":             len(records),
            "iter":                it,
            "incumbent_full_eval": float(incumbent_full_score),
            "full_eval":           float(full_score),
            "full_eval_delta":     float(full_score) - float(incumbent_full_score),
            "ls_steps_taken":      len(proxy_scores) - 1,  # steps actually executed
        }
        for k in range(LS_MOVES + 1):
            record[f"proxy_{k}"] = float(proxy_scores[k]) if k < len(proxy_scores) else float("nan")
        records.append(record)

        if len(records) % 100 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  {len(records):4d}/{N_TRAJ}  iter={it}  "
                f"best_full={best_full_score:.4f}  elapsed={elapsed:.1f}s"
            )

    dispose_eval()

    elapsed_total = time.perf_counter() - t0
    print(f"\nDone. {len(records)} trajectories in {elapsed_total:.1f}s  ({it} DR iters)")

    # -------------------------
    # Save CSV
    # -------------------------
    df = pd.DataFrame(records)
    csv_out = out_dir / "trajectories.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved: {csv_out}")

    proxy_cols = [f"proxy_{k}" for k in range(LS_MOVES + 1)]

    # -------------------------
    # ANALYSIS
    # -------------------------
    print("\n===== ANALYSIS =====")
    print(df[["full_eval", "full_eval_delta", "ls_steps_taken"] + proxy_cols].describe().to_string())

    # Early-stop distribution
    print("\nLS steps-taken distribution (hill-climbing early stop):")
    print(df["ls_steps_taken"].value_counts().sort_index().to_string())

    steps   = list(range(LS_MOVES + 1))
    fe      = df["full_eval"].to_numpy(dtype=float)
    fe_min, fe_max = fe.min(), fe.max()
    fe_norm = (fe - fe_min) / max(fe_max - fe_min, 1e-12)

    delta   = df["full_eval_delta"].to_numpy(dtype=float)

    def _pearson_dropna(col_vals, target):
        """Pearson r ignoring rows where col_vals is NaN (early-stopped trajectories)."""
        mask = ~np.isnan(col_vals)
        if mask.sum() < 10:
            return float("nan")
        return float(np.corrcoef(col_vals[mask], target[mask])[0, 1])

    # --- Pearson r(proxy_k, full_eval)  [absolute quality] ---
    corrs_abs   = [_pearson_dropna(df[f"proxy_{k}"].to_numpy(dtype=float), fe)    for k in steps]
    # --- Pearson r(proxy_k, full_eval_delta)  [improvement over incumbent] ---
    corrs_delta = [_pearson_dropna(df[f"proxy_{k}"].to_numpy(dtype=float), delta) for k in steps]

    print("\nPearson r(proxy_k, full_eval)          r(proxy_k, full_eval_delta):")
    for k, (ra, rd) in enumerate(zip(corrs_abs, corrs_delta)):
        print(f"  step {k:2d}:  abs={ra:+.4f}   delta={rd:+.4f}")

    # Proxy-improvement vs full_eval_delta: does LS proxy gain predict true gain?
    # Use the last non-NaN proxy value per row (the plateau the search settled at)
    proxy_final_actual = df[proxy_cols].apply(
        lambda row: row.dropna().iloc[-1] if row.dropna().shape[0] > 0 else float("nan"),
        axis=1
    ).to_numpy(dtype=float)
    proxy_improvement = proxy_final_actual - df["proxy_0"].to_numpy(dtype=float)

    # Correlation of last-accepted proxy with full_eval (used in fig1 panel 3)
    _mask_final = ~np.isnan(proxy_final_actual)
    r_proxy_final_fe = float(np.corrcoef(proxy_final_actual[_mask_final], fe[_mask_final])[0, 1]) \
        if _mask_final.sum() >= 2 else float("nan")
    _mask_imp = ~np.isnan(proxy_improvement)
    r_imp_delta = float(np.corrcoef(proxy_improvement[_mask_imp], delta[_mask_imp])[0, 1])
    print(f"\nr(proxy_improvement, full_eval_delta) = {r_imp_delta:+.4f}")
    print("  (proxy improvement = proxy_final - proxy_0 within one LS run)")

    frac_positive_delta = float((delta > 0).mean())
    print(f"\nFraction of trajectories that improved over incumbent: {frac_positive_delta:.3f}")
    # Before the regression block, add:
    # Only use proxy columns with enough non-NaN rows
    valid_cols = [c for c in proxy_cols if df[c].notna().sum() >= 10]
    if len(valid_cols) == 0:
        print("  No usable proxy columns -- skipping regression and plots.")
        return
    X_df = df[valid_cols].copy()
    for col in valid_cols:
        X_df[col] = X_df[col].fillna(df["proxy_0"])
    X = X_df.values
    if X.shape[0] == 0:
        print("  No complete trajectories -- skipping regression.")
        return
    # -------------------------
    # REGRESSION  (two targets: absolute and delta)
    # -------------------------
    print("\n----- Regression -----")
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings("ignore")

    # Only use proxy columns that have enough non-NaN rows (early stop makes later cols sparse)
    MIN_SAMPLES = max(10, int(0.01 * len(df)))
    active_proxy_cols = [c for c in proxy_cols if df[c].notna().sum() >= MIN_SAMPLES]
    print(f"\nProxy columns with >={MIN_SAMPLES} non-NaN rows: {active_proxy_cols}")

    df_full = df[active_proxy_cols + ["full_eval", "full_eval_delta"]].dropna()
    print(f"Rows usable for regression: {len(df_full)}/{len(df)}")

    if len(df_full) < 10:
        print("  Too few rows for regression -- skipping.")
    else:
        X       = df_full[active_proxy_cols].to_numpy(dtype=float)
        scaler  = StandardScaler()
        X_s     = scaler.fit_transform(X)
        X_final = df_full[[active_proxy_cols[-1]]].to_numpy(dtype=float)

        fe_full    = df_full["full_eval"].to_numpy(dtype=float)
        delta_full = df_full["full_eval_delta"].to_numpy(dtype=float)

        for target_name, y_target in [
            ("full_eval       (absolute)", fe_full),
            ("full_eval_delta (improvement)", delta_full),
        ]:
            print(f"\n  Target: {target_name}")
            for name, clf, X_fit in [
                ("LinearRegression  -- all active steps",  LinearRegression(),                                      X_s),
                ("LinearRegression  -- final active step", LinearRegression(),                                      X_final),
                ("RandomForest      -- all active steps",  RandomForestRegressor(n_estimators=200, random_state=0), X_s),
                ("GradientBoosting  -- all active steps",  GradientBoostingRegressor(n_estimators=200, random_state=0), X_s),
            ]:
                cv_r2 = cross_val_score(clf, X_fit, y_target, cv=5, scoring="r2")
                print(f"    {name:50s}  CV R² = {cv_r2.mean():+.4f} ± {cv_r2.std():.4f}")

        for target_name, y_target in [
            ("full_eval", fe_full),
            ("full_eval_delta", delta_full),
        ]:
            rf = RandomForestRegressor(n_estimators=200, random_state=0)
            rf.fit(X_s, y_target)
            importances = rf.feature_importances_
            print(f"\nRF importances -> {target_name}:")
            for col, imp in zip(active_proxy_cols, importances):
                print(f"  {col}: {imp:.4f}")

    # -------------------------
    # PLOTS
    # -------------------------
    # Only show steps that actually have data
    max_actual_step = int(df["ls_steps_taken"].max())
    plot_steps = list(range(max_actual_step + 1))

    # Colour helpers
    delta_min, delta_max = delta.min(), delta.max()
    delta_norm = (delta - delta_min) / max(delta_max - delta_min, 1e-12)

    # -----------------------------------------------------------------------
    # Figure 1 -- absolute quality (full_eval)
    # -----------------------------------------------------------------------
    fig1, axes = plt.subplots(2, 2, figsize=(14, 10))
    eval_label = (
        f"UE (mu={args.ue_mu}, noopt={args.ue_noopt}km, "
        f"range={args.ue_max_range}km, α_w={args.ue_alpha_wait}, pd={args.ue_penalty_distance})"
        if use_ue else "GRB"
    )
    fig1.suptitle(
        f"Proxy-trajectory study -- absolute quality  |  {len(records)} trajectories  |  {args.csv}  |  full_eval={eval_label}",
        fontsize=11
    )

    # Panel 1: trajectory bundles coloured by full_eval
    ax = axes[0, 0]
    cmap = plt.cm.RdYlGn
    idx_sample = np.random.default_rng(0).choice(len(df), size=min(200, len(df)), replace=False)
    for idx in idx_sample:
        row = df.iloc[idx]
        raw  = [row[f"proxy_{k}"] for k in plot_steps if not np.isnan(row[f"proxy_{k}"])]
        xs   = [k for k in plot_steps if not np.isnan(row[f"proxy_{k}"])]
        traj = list(np.maximum.accumulate(raw))   # running max: LS only accepts improvements
        ax.plot(xs, traj, color=cmap(fe_norm[idx]), alpha=0.25, linewidth=0.8)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(fe_min, fe_max))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="full_eval")
    ax.set_xlabel("LS step")
    ax.set_ylabel("proxy score")
    ax.set_title("Proxy trajectories (colour = final full-eval)")
    ax.set_xticks(plot_steps)

    # Panel 2: r(proxy_k, full_eval) per step
    ax = axes[0, 1]
    corrs_abs_plot = corrs_abs[:max_actual_step + 1]
    ax.bar(plot_steps, corrs_abs_plot, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LS step k")
    ax.set_ylabel("Pearson r(proxy_k, full_eval)")
    ax.set_title("Absolute predictive power vs step")
    ax.set_xticks(plot_steps)
    ax.set_ylim(min(0, float(np.nanmin(corrs_abs_plot))) - 0.05, 1.05)

    # Panel 3: scatter proxy_final (last accepted) vs full_eval
    ax = axes[1, 0]
    ax.scatter(proxy_final_actual, fe, alpha=0.3, s=8, color="steelblue")
    ax.set_xlabel("proxy_final  (last accepted LS proxy)")
    ax.set_ylabel("full_eval")
    ax.set_title(f"proxy_final vs full_eval  (r={r_proxy_final_fe:.3f})")

    # Panel 4: scatter proxy_0 vs full_eval (D&R output alone)
    ax = axes[1, 1]
    ax.scatter(df["proxy_0"], fe, alpha=0.3, s=8, color="darkorange")
    ax.set_xlabel("proxy_0  (after D&R, before LS)")
    ax.set_ylabel("full_eval")
    ax.set_title(f"proxy_0 vs full_eval  (r={corrs_abs[0]:.3f})")

    fig1.tight_layout()
    fig1_path = out_dir / "trajectory_analysis_absolute.png"
    fig1.savefig(fig1_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved: {fig1_path}")

    # -----------------------------------------------------------------------
    # Figure 2 -- delta analysis (improvement over incumbent)
    # -----------------------------------------------------------------------
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(
        f"Proxy-trajectory study -- improvement over incumbent  |  {len(records)} trajectories  |  {args.csv}  |  full_eval={eval_label}",
        fontsize=11
    )

    # Panel 1: trajectory bundles coloured by full_eval_delta
    ax = axes2[0, 0]
    cmap_div = plt.cm.RdYlGn
    for idx in idx_sample:
        row = df.iloc[idx]
        raw  = [row[f"proxy_{k}"] for k in plot_steps if not np.isnan(row[f"proxy_{k}"])]
        xs   = [k for k in plot_steps if not np.isnan(row[f"proxy_{k}"])]
        traj = list(np.maximum.accumulate(raw))   # running max: LS only accepts improvements
        ax.plot(xs, traj, color=cmap_div(delta_norm[idx]), alpha=0.25, linewidth=0.8)
    sm2 = plt.cm.ScalarMappable(cmap=cmap_div, norm=plt.Normalize(delta_min, delta_max))
    sm2.set_array([])
    plt.colorbar(sm2, ax=ax, label="full_eval_delta")
    ax.set_xlabel("LS step")
    ax.set_ylabel("proxy score")
    ax.set_title("Proxy trajectories (colour = Delta over incumbent)")
    ax.set_xticks(plot_steps)

    # Panel 2: r(proxy_k, full_eval_delta) per step
    ax = axes2[0, 1]
    corrs_delta_plot = corrs_delta[:max_actual_step + 1]
    colors = ["steelblue" if (not np.isnan(r) and r >= 0) else "firebrick" for r in corrs_delta_plot]
    ax.bar(plot_steps, corrs_delta_plot, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("LS step k")
    ax.set_ylabel("Pearson r(proxy_k, full_eval_delta)")
    ax.set_title("Does proxy_k predict improvement over incumbent?")
    ax.set_xticks(plot_steps)
    lo = min(float(np.nanmin(corrs_delta_plot)) - 0.05, -0.1)
    ax.set_ylim(lo, max(float(np.nanmax(corrs_delta_plot)) + 0.05, 0.2))

    # Panel 3: proxy improvement (proxy_final - proxy_0) vs full_eval_delta
    ax = axes2[1, 0]
    ax.scatter(proxy_improvement[_mask_imp], delta[_mask_imp], alpha=0.3, s=8, color="mediumseagreen")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("proxy improvement within LS  (proxy_final - proxy_0)")
    ax.set_ylabel("full_eval_delta  (full_eval - incumbent)")
    ax.set_title(f"Proxy gain vs true gain  (r={r_imp_delta:+.3f})")

    # Panel 4: proxy_0 vs full_eval_delta
    ax = axes2[1, 1]
    ax.scatter(df["proxy_0"], delta, alpha=0.3, s=8, color="darkorange")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("proxy_0  (after D&R, before LS)")
    ax.set_ylabel("full_eval_delta")
    ax.set_title(f"Initial proxy vs improvement over incumbent  (r={corrs_delta[0]:+.3f})")

    fig2.tight_layout()
    fig2_path = out_dir / "trajectory_analysis_delta.png"
    fig2.savefig(fig2_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {fig2_path}")

    if SHOW_PLOT:
        plt.show()

    print("\n===== DONE =====")


if __name__ == "__main__":
    main()
