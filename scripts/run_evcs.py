# =========================
# 0) IMPORTS & PATHS
# =========================
from pathlib import Path
import time
import pandas as pd
import sys
import numpy as np
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from evcs import (
    build_multi_period_model,
    solve_model,
    run_DR_multi
)

from evcs.methods import sync_solution_state, reassign_y_greedy_multi
from evcs.geom import build_arcs
import matplotlib
matplotlib.use("Agg")   # required on Narval (no display)
import matplotlib.pyplot as plt


# =========================
# ARGUMENTS
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--csv",  type=str,   default="center_102_Vicenza_k125.csv")
parser.add_argument("--seed", type=int,   default=11)
parser.add_argument("--D",    type=float, default=0.5)
parser.add_argument("--T",    type=int,   default=3)
parser.add_argument("--Q",    type=float, default=20.0)
args = parser.parse_args()

# =========================
# CONFIG
# =========================
DATA_DIR  = PROJECT_ROOT / "data" / "input"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_NAME              = args.csv
D                     = args.D
T                     = args.T
Q                     = args.Q
seed                  = args.seed

policy                = "closest_priority"
max_chargers_per_site = 6

# DR parameters
max_iter       = 2000     # high so time limit always triggers first
dr_time_limit  = 600      # 10 minutes
batch_size     = 50
top_k_full     = 8
ls_moves       = 12
ls_frac_remove = 0.08
frac_remove    = 0.3
accept_epsilon = 1e-6
adaptive_destroy = True
destroy_modes  = ["site_swap", "local_remove", "area_destroy"]

# Exact solver
exact_time_limit = 300    # 5 minutes (enough for small instances)
mip_gap          = 0.001


# =========================
# INSTANCE BUILDER
# Identical to notebook build_inst_from_csv()
# =========================
def build_instance_from_csv(csv_path, T, seed, D_km):
   
    df = pd.read_csv(csv_path)

    # coordinates degrees -> km
    coords_deg = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(dtype=float)
    coords_km = coords_deg.copy()
    coords_km[:, 0] *= 111.0
    coords_km[:, 1] *= 111.0

    # demand from population share
    pop = df["Aggregated_Population"].to_numpy(dtype=float)
    pop = np.maximum(pop, 0.0)
    if pop.sum() <= 0:
        raise ValueError("Population column sums to zero.")
    pop_share = pop / pop.sum()

    M = coords_km.shape[0]
    base = pop_share * M   # normalized so sum(base) = M

    # seasonal demand with noise — identical to notebook
    rng = np.random.default_rng(seed)
    demand_IT = np.zeros((T, M), dtype=float)
    for t in range(T):
        season_factor = 1.0 + 0.2 * np.sin(2 * np.pi * t / T)
        noise = rng.normal(0.0, 0.05, size=M)
        demand_IT[t] = np.maximum(0.0, base * season_factor * (1.0 + noise))

    # build arcs — forbid_self=False same as notebook
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
def run_single_experiment():
    global T, seed

    print("Starting EVCS experiment")
    t_global_start = time.time()

    # -------------------------
    # LOAD — identical to notebook
    # -------------------------
    inst = build_instance_from_csv(
        DATA_DIR / CSV_NAME, T=T, seed=seed, D_km=D
    )

    M         = inst["M"]
    N         = inst["N"]
    distIJ    = inst["distIJ"]
    in_range  = inst["in_range"]
    Ji        = inst["Ji"]
    Ij        = inst["Ij"]
    demand_IT = inst["demand_IT"]   # shape (T, M)
    P_T_local = [8] * T             # budget per period

    # -------------------------
    # DIAGNOSTICS — compare with notebook
    # -------------------------
    print("=== SCRIPT DIAGNOSTICS ===")
    print(f"CSV              : {CSV_NAME}")
    print(f"M (demand nodes) : {M}")
    print(f"N (sites)        : {N}")
    print(f"T (periods)      : {T}")
    print(f"Arcs in_range    : {len(in_range)}")
    print(f"Demand period 0  : {demand_IT[0].sum():.4f}")
    print(f"Demand period 1  : {demand_IT[1].sum():.4f}")
    print(f"Demand period 2  : {demand_IT[2].sum():.4f}")
    print(f"Total demand     : {demand_IT.sum():.4f}")
    print(f"Min demand node  : {demand_IT.min():.4f}")
    print(f"Max demand node  : {demand_IT.max():.4f}")
    print(f"D (km)           : {D}")
    print(f"Q                : {Q}")
    print(f"P_T              : {P_T_local}")
    print(f"seed             : {seed}")
    print(f"Total capacity   : {Q * sum(P_T_local):.1f}")
    print("==========================")

    # -------------------------
    # DR
    # -------------------------
    print("\n Running DR...")
    t_dr_start = time.time()

    dr_out = run_DR_multi(
        inst=inst,
        policy=policy,
        P_T=P_T_local,
        Q=Q,
        D=D,
        T=T,
        max_iter=max_iter,
        dr_time_limit=dr_time_limit,
        seed=seed,
        max_chargers_per_site=max_chargers_per_site,
        frac_remove=frac_remove,
        accept_epsilon=accept_epsilon,
        adaptive_destroy=adaptive_destroy,
        destroy_modes=destroy_modes,
        batch_size=batch_size,
        top_k_full=top_k_full,
        ls_moves=ls_moves,
        ls_frac_remove=ls_frac_remove,
    )

    t_dr_end = time.time()

    dr_best = dr_out["best_obj"]
    print(f"DR best = {dr_best:.4f}")

    # DR debug info
    U_best = dr_out["U_best"]
    total_chargers = sum(U_best.values())
    total_capacity = total_chargers * Q
    print(f"  chargers installed : {total_chargers}")
    print(f"  total capacity     : {total_capacity:.1f}")
    print(f"  total demand       : {demand_IT.sum():.1f}")
    print(f"  max coverable      : {min(total_capacity, demand_IT.sum()):.1f}")

    # -------------------------
    # EXACT
    # -------------------------
    print("\n Running Exact...")
    t_exact_start = time.time()

    model = build_multi_period_model(
        M=M, N=N, T=T,
        in_range=in_range,
        Ji=Ji,
        Ij=Ij,
        demand_IT=demand_IT,
        Q=Q,
        P_T=P_T_local,
        distIJ=distIJ,
        method_name=policy,
        max_chargers_per_site=max_chargers_per_site,
    )

    res = solve_model(
        model,
        time_limit=exact_time_limit,
        mip_gap=mip_gap,
        solver_name="gurobi",
    )

    t_exact_end = time.time()

    exact_inc_raw  = getattr(res, "best_feasible_objective", None)
    exact_bound_raw = getattr(res, "best_objective_bound", None)

    if exact_inc_raw is not None:
        exact_inc_raw = float(exact_inc_raw)
    if exact_bound_raw is not None:
        exact_bound_raw = float(exact_bound_raw)

    if exact_inc_raw is not None and exact_bound_raw is not None:
        exact_gap_raw = abs(exact_inc_raw - exact_bound_raw) / max(abs(exact_inc_raw), 1e-9)
    else:
        exact_gap_raw = None

    # -------------------------
    # Evaluate EXACT with greedy assignment
    # (same evaluation method as DR uses)
    # Only run if Gurobi actually found a feasible integer solution
    # -------------------------
    exact_obj = None
    exact_has_feasible = (
        exact_inc_raw is not None
        and any((model.u[j, t].value or 0) > 0.5 for j in model.J for t in model.T)
    )

    if exact_has_feasible:
        U_exact = {
            (j, t): int(round(model.u[j, t].value))
            for j in model.J
            for t in model.T
        }

        m_tmp = model.clone()
        for (j, t), val in U_exact.items():
            m_tmp.u[j, t].value = val

        sync_solution_state(m_tmp, cumulative_install=True)
        reassign_y_greedy_multi(
            m_tmp, distIJ,
            Ji=inst["Ji"],
            method_name=policy,
            cumulative_install=True,
        )

        exact_obj = 0.0
        for t in m_tmp.T:
            for i in m_tmp.I:
                for j in inst["Ji"][i]:
                    val = m_tmp.y[i, j, t].value
                    if val is not None and val > 0.5:
                        exact_obj += demand_IT[t, i]
                        break
        print(f"Exact obj = {exact_obj:.4f}")
    else:
        print("Exact: no feasible integer solution found within time limit.")

    # -------------------------
    # RESULTS
    # -------------------------
    t_global_end = time.time()

    if exact_obj is not None and exact_obj > 1e-9:
        gap_abs = exact_obj - dr_best
        gap_pct = 100 * gap_abs / exact_obj
    else:
        gap_abs = None
        gap_pct = None

    print("\n===== RESULTS =====")
    print(f"Instance : {CSV_NAME}")
    print(f"N, M, T  : {N}, {M}, {T}")
    print(f"Policy   : {policy}")
    print(f"seed     : {seed}")
    print(f"DR best  : {dr_best:.4f}")
    print(f"Exact    : {exact_obj:.4f}" if exact_obj is not None else "Exact    : N/A (no feasible solution)")
    print(f"Gap      : {gap_abs:.4f} ({gap_pct:.2f}%)" if gap_pct is not None else "Gap      : N/A")
    print(f"Time DR  : {t_dr_end - t_dr_start:.2f}s")
    print(f"Time EX  : {t_exact_end - t_exact_start:.2f}s")
    print(f"Time TOT : {t_global_end - t_global_start:.2f}s")
    print("=====================\n")

    # -------------------------
    # EXCEL LOG
    # -------------------------
    excel_file = RESULTS_DIR / "benchmark_new_algorithm.xlsx"

    exact_aligned    = exact_obj  # None if no feasible integer solution
    dr_aligned       = dr_best
    gap_aligned      = (exact_aligned - dr_aligned) if exact_aligned is not None else None
    gap_aligned_pct  = (100 * gap_aligned / exact_aligned) if (gap_aligned is not None and exact_aligned > 1e-9) else None

    gap_raw_inc      = None
    gap_raw_inc_pct  = None
    if exact_inc_raw is not None:
        gap_raw_inc     = exact_inc_raw - dr_aligned
        gap_raw_inc_pct = 100 * gap_raw_inc / max(abs(exact_inc_raw), 1e-9)

    row = {
        "Instance":             Path(CSV_NAME).stem,
        "Policy":               policy,
        "N":                    N,
        "M":                    M,
        "T":                    T,
        "seed":                 seed,
        "D":                    D,
        "Q":                    Q,
        # EXACT
        "Exact_aligned":        exact_aligned,
        "Exact_incumbent_raw":  exact_inc_raw,
        "Exact_bound_raw":      exact_bound_raw,
        "Exact_gap_raw":        exact_gap_raw,
        # DR
        "DR_aligned":           dr_aligned,
        "DR_iters":             len(dr_out.get("DR_trace", [])),
        # GAPS
        "Gap_aligned":          gap_aligned,
        "Gap_aligned_%":        gap_aligned_pct,
        "Gap_raw_inc":          gap_raw_inc,
        "Gap_raw_inc_%":        gap_raw_inc_pct,
        # TIME
        "DR_time":              t_dr_end - t_dr_start,
        "Exact_time":           t_exact_end - t_exact_start,
        "Total_time":           t_global_end - t_global_start,
    }

    df_new = pd.DataFrame([row])
    cols   = list(row.keys())
    df_new = df_new[cols]

    if excel_file.exists():
        try:
            df_old = pd.read_excel(excel_file)
            for col in df_new.columns:
                if col not in df_old.columns:
                    df_old[col] = None
            for col in df_old.columns:
                if col not in df_new.columns:
                    df_new[col] = None
            df_new = df_new[df_old.columns]
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            print(f"⚠️ Could not read Excel, creating new: {e}")
            df_all = df_new
    else:
        df_all = df_new

    df_all.to_excel(excel_file, index=False)
    print(f"📊 Excel updated → {excel_file}")

    # -------------------------
    # SAVE TRACE CSV for local plotting
    # (supervisor: generate plots on your machine, not server)
    # -------------------------
    trace = dr_out.get("DR_trace")
    if trace is not None and len(trace) > 0:
        trace_path = RESULTS_DIR / f"trace_{Path(CSV_NAME).stem}_seed{seed}.csv"
        trace.to_csv(trace_path, index=False)
        print(f"📈 Trace saved → {trace_path}")
        print("   (generate plot locally from this CSV)")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run_single_experiment()