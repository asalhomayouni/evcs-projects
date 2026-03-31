# =========================
# 0) IMPORTS & PATHS
# =========================
from pathlib import Path
import time
import pandas as pd
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from evcs import (
    load_instance,
    build_multi_period_model,
    solve_model,
    run_DR_multi
)
 
from evcs.methods import sync_solution_state, reassign_y_greedy_multi
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
DATA_DIR = PROJECT_ROOT / "data" / "input"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CSV_NAME = "center_240_Parma_k200.csv"
D =0.02
T = 3
P_T = [3,3,3]
policy = "closest_priority"
Q = 500
max_chargers_per_site = 10

# DR
max_iter = 30
dr_time_limit = 600
seed = 11
batch_size = 20   

frac_remove = 0.3
accept_epsilon = 0.01
adaptive_destroy = True
destroy_modes = ["site_swap", "local_remove", "area_destroy"]

# Exact
<<<<<<< HEAD
exact_time_limit = 1800
=======
exact_time_limit = 600
>>>>>>> eb6485441a2810d973361936d1effee409ecb622
mip_gap = 1e-4


# =========================
# MAIN
# =========================
def run_single_experiment():

    print("Starting EVCS experiment")
    t_global_start = time.time()

    # -------------------------
    # LOAD
    # -------------------------
    inst = load_instance(DATA_DIR / CSV_NAME, radius=D)

    M, N = inst["M"], inst["N"]
    Ji, Ij = inst["Ji"], inst["Ij"]
    distIJ = inst["distIJ"]
    in_range = inst["in_range"]
    demand_IT = inst["demand_IT"]

   
    # Demand handling (ROBUST)
    # -------------------------
    demand_IT = np.asarray(inst["demand_IT"], dtype=float)

    # Fix orientation
    if demand_IT.shape[0] == M:
        demand_IT = demand_IT.T

    # T from data
    T_data = demand_IT.shape[0]
<<<<<<< HEAD
    T = min(T, T_data)   # use CONFIG T=4, not data T=8
    demand_IT = demand_IT[:T, :] 

=======
	T = min(T, T_data)
	demand_IT = demand_IT[:T, :]
>>>>>>> eb6485441a2810d973361936d1effee409ecb622
    # ✅ MAKE LOCAL COPY (IMPORTANT)
    P_T_local = list(P_T)

    # Align P_T safely
    if len(P_T_local) > T:
        P_T_local = P_T_local[:T]
    elif len(P_T_local) < T:
        P_T_local = P_T_local + [P_T_local[-1]] * (T - len(P_T_local))

    inst["demand_IT"] = demand_IT

    print("FINAL SHAPE:", demand_IT.shape)
    print("Using T =", T)
    # -------------------------
    # DR
    # -------------------------
    print(" Running DR...")
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
    )

    t_dr_end = time.time()

    dr_best = dr_out["best_obj"]
    print(f"DR best = {dr_best:.2f}")

    # -------------------------
    # EXACT
    # -------------------------
    print(" Running Exact...")
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
        max_chargers_per_site=max_chargers_per_site
    )

    res = solve_model(
        model,
        time_limit=exact_time_limit,
        mip_gap=mip_gap,
        solver_name="gurobi"
    )

    t_exact_end = time.time()

    exact_inc_raw = getattr(res, "best_feasible_objective", None)
    exact_bound_raw = getattr(res, "best_objective_bound", None)

    if exact_inc_raw is not None:
        exact_inc_raw = float(exact_inc_raw)

    if exact_bound_raw is not None:
        exact_bound_raw = float(exact_bound_raw)

    # compute raw gap
    if exact_inc_raw is not None and exact_bound_raw is not None:
        exact_gap_raw = abs(exact_inc_raw - exact_bound_raw) / max(abs(exact_inc_raw), 1e-9)
    else:
        exact_gap_raw = None

    # -------------------------
    # Extract U
    # -------------------------
    U_exact = {
        (j, t): int(round(model.u[j, t].value))
        for j in model.J
        for t in model.T
    }

    # -------------------------
    # Evaluate EXACT (same as DR)
    # -------------------------
    m_tmp = model.clone()

    for (j, t), val in U_exact.items():
        m_tmp.u[j, t].value = val

    sync_solution_state(m_tmp, cumulative_install=True)

    reassign_y_greedy_multi(
        m_tmp,
        distIJ,
        Ji=inst["Ji"],
        method_name=policy,
        cumulative_install=True
    )

    exact_obj = 0.0
    for t in m_tmp.T:
        for i in m_tmp.I:
            for j in inst["Ji"][i]:
                val = m_tmp.y[i, j, t].value
                if val is not None and val > 0.5:
                    exact_obj += demand_IT[t, i]
                    break

    print(f"Exact obj = {exact_obj:.2f}")

    # SAVE + REPORT (CLEAN)

    t_global_end = time.time()

    # ---- Summary numbers ----
    gap_abs = exact_obj - dr_best
    gap_pct = 100 * gap_abs / max(exact_obj, 1e-9)
    gap_rel = gap_abs / max(exact_obj, 1e-9)   # fractional gap for logging
    total_demand = float(demand_IT.sum())

    # =========================
    # TERMINAL OUTPUT
    # =========================
    print("\n===== RESULTS =====")
    print(f"Instance : {CSV_NAME}")
    print(f"N, M, T  : {N}, {M}, {T}")
    print(f"Policy   : {policy}")
    print(f"DR best  : {dr_best:.4f}")
    print(f"Exact    : {exact_obj:.4f}")
    print(f"Gap      : {gap_abs:.4f} ({gap_pct:.2f}%)")
    print(f"Time DR  : {t_dr_end - t_dr_start:.2f}s")
    print(f"Time EX  : {t_exact_end - t_exact_start:.2f}s")
    print("=====================\n")

    # =========================
    # EXCEL LOG (RESEARCH LEVEL)
    # =========================
    excel_file = RESULTS_DIR / "benchmark_new_algorithm.xlsx"

    total_demand = float(demand_IT.sum())

    # -------------------------
    # ALIGNED OBJECTIVES
    # -------------------------
    exact_aligned = exact_obj
    dr_aligned = dr_best

    gap_aligned = exact_aligned - dr_aligned
    gap_aligned_pct = 100 * gap_aligned / max(exact_aligned, 1e-9)

    # -------------------------
    # RAW OBJECTIVES (MIP)
    # -------------------------
    gap_raw_inc = None
    gap_raw_inc_pct = None

    if exact_inc_raw is not None:
        # DR has no true raw → compare to aligned proxy (optional)
        dr_raw_proxy = dr_aligned

        gap_raw_inc = exact_inc_raw - dr_raw_proxy
        gap_raw_inc_pct = 100 * gap_raw_inc / max(abs(exact_inc_raw), 1e-9)

    # -------------------------
    # BUILD ROW
    # -------------------------
    row = {
        "Instance": Path(CSV_NAME).stem,
        "Policy": policy,
        "N": N,
        "M": M,
        "T": T,
        "seed": seed,

        # ===== EXACT =====
        "Exact_aligned": exact_aligned,
        "Exact_incumbent_raw": exact_inc_raw,
        "Exact_bound_raw": exact_bound_raw,
        "Exact_gap_raw": exact_gap_raw,

        # ===== DR =====
        "DR_aligned": dr_aligned,
        "DR_iters": len(dr_out.get("DR_trace", [])),

        # ===== GAPS =====
        "Gap_aligned": gap_aligned, 
        "Gap_aligned_%": gap_aligned_pct,

        "Gap_raw_inc": gap_raw_inc,
        "Gap_raw_inc_%": gap_raw_inc_pct,

        # ===== TIME =====
        "DR_time": t_dr_end - t_dr_start,
        "Exact_time": t_exact_end - t_exact_start,
        "Total_time": t_global_end - t_global_start,
    }

    df_new = pd.DataFrame([row])

    # enforce consistent columns
    cols = list(row.keys())
    df_new = df_new[cols]

    # -------------------------
    # APPEND SAFELY
    # -------------------------
    if excel_file.exists():
        try:
            df_old = pd.read_excel(excel_file)

            # 🔥 FORCE SAME COLUMNS WITHOUT RESETTING FILE
            for col in df_new.columns:
                if col not in df_old.columns:
                    df_old[col] = None

            for col in df_old.columns:
                if col not in df_new.columns:
                    df_new[col] = None

            # reorder columns
            df_new = df_new[df_old.columns]

            df_all = pd.concat([df_old, df_new], ignore_index=True)

        except Exception as e:
            print("⚠️ Could not read Excel, creating new one:", e)
            df_all = df_new
    else:
        df_all = df_new

    df_all.to_excel(excel_file, index=False)

    print(f"📊 Updated Excel → {excel_file}")
   
    # DR vs EXACT PLOT (FIXED)
    # =========================
    import matplotlib
    matplotlib.use("Agg")  # 🔥 REQUIRED on Narval

    trace = dr_out.get("DR_trace")

    if trace is not None and len(trace) > 0:

        x = trace["iteration"].to_numpy()

        # current (fluctuating)
        if "current" in trace.columns:
            current = trace["current"].to_numpy()
        else:
            current = trace["proxy_curr"].to_numpy()

        # best-so-far
        if "best_full" in trace.columns:
            best = trace["best_full"].ffill().to_numpy()
        else:
            best = trace["best"].ffill().to_numpy()

        plt.figure(figsize=(8, 5))

        plt.plot(x, current, alpha=0.4, label="DR fluctuating (current)")
        plt.plot(x, best, linewidth=2, label="DR best-so-far (best_full)")

        plt.axhline(
            y=exact_obj,
            linestyle="--",
            linewidth=2,
            label=f"Exact ({exact_obj:.3f})"
        )

        plt.xlabel("iteration")
        plt.ylabel("Score")

        plt.title(f"DR vs Exact | N={N}, T={T}, seed={seed} | policy={policy}")

        plt.grid(True)
        plt.legend()

        plot_path = RESULTS_DIR / f"dr_curve_{Path(CSV_NAME).stem}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")

        print(f"📈 Plot saved → {plot_path}")

        plt.close()
# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    run_single_experiment()
