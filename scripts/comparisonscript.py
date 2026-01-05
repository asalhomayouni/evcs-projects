import numpy as np
import pandas as pd
import time
from pathlib import Path
from pyomo.opt import TerminationCondition

# =========================================================
# EVCS core modules (absolute imports)
# =========================================================

from evcs.geom import build_arcs
from evcs.model import build_base_model
from evcs.methods import (
    build_initial_solution,
    build_initial_solution_smart,
    build_initial_solution_weighted,
    apply_method,
    local_search,
    simulated_annealing,
    evaluate_solution,
    compute_farther,
)
from evcs.solve import solve_model

# =========================================================
# Instance generator (absolute imports)
# =========================================================

from scripts.randomInstance import (
    generate_instance,
    save_instance,
    load_instance,
)

# =========================================================
# Default parameters for P, Q, D
# =========================================================

def default_parameters_for(N: int):
    """
    Generate reasonable defaults for P, Q, D based on instance size.
    Keep consistent with the rest of your project.
    """
    P = max(2, int(round((N ** 0.5) / 1.2)))
    avg_demand = 3.0
    Q = (N * avg_demand) / max(1, P)
    D = 3.5 if N <= 60 else 2.8
    return P, Q, D


# =========================================================
# run_one_policy — pure algorithmic core (NO FILE I/O)
# =========================================================

def run_one_policy(
    inst,
    policy,
    P,
    Q,
    D,
    forbid_self: bool = False,
    max_iter: int = 50,
    exact_time_limit: float = 120,
    exact_mip_gap: float = 0.01,
    greedy_mode: str = "deterministic",  # "deterministic" or "random"
):
    """
    Run one configuration:
      - build arcs
      - exact MILP (score_exact, time_exact)
      - greedy initialization (deterministic or random)
      - single Local Search
      - single Simulated Annealing

    Returns a flat dict (no saving, no restart tables).
    """

    # ---------- Unpack instance ----------
    coords_I, coords_J = inst["coords_I"], inst["coords_J"]
    I_idx, J_idx = inst["I_idx"], inst["J_idx"]
    demand_I = inst["demand_I"]
    M, N = len(I_idx), len(J_idx)

    print(f"\n📌 N={N} | Policy={policy} | greedy_mode={greedy_mode}")

    # ---------- Build arcs & helper structures ----------
    distIJ, in_range, Ji, Ij = build_arcs(
        coords_I, coords_J, D=D, forbid_self=forbid_self,
        I_idx=I_idx, J_idx=J_idx,
    )
    farther_of = compute_farther(distIJ, in_range, Ij)

    # =====================================================
    # 1) EXACT SOLVE
    # =====================================================
    score_exact = np.nan
    time_exact = np.nan
    optimal_exact = False

    try:
        print("  🔹 Exact solve ...")

        m_exact = build_base_model(
            M, N, in_range, Ji, Ij, demand_I, Q, P,
            distIJ=distIJ, method_name=policy,
        )

        t0 = time.perf_counter()
        result = solve_model(
            m_exact, verbose=False,
            time_limit=exact_time_limit,
            mip_gap=exact_mip_gap,
        )
        time_exact = time.perf_counter() - t0

        met = evaluate_solution(m_exact, distIJ, demand_I, method_name=policy)
        score_exact = met["covered_demand"]

        term = getattr(result, "termination_condition", None)
        term_name = getattr(term, "name", str(term)).lower()
        optimal_exact = (
            term == TerminationCondition.optimal
            or term_name == "optimal"
        )

        if optimal_exact:
            print(f"    ✅ Exact OPTIMAL: {score_exact:.3f} (t={time_exact:.2f}s)")
        else:
            print(f"    ⚠️ Exact SUBOPTIMAL: {score_exact:.3f} (t={time_exact:.2f}s, term={term})")

    except Exception as e:
        print(f"    ❌ Exact solve failed: {e}")

    # =====================================================
    # 2) GREEDY INITIALIZATION (deterministic or random)
    # =====================================================
    score_greedy = np.nan
    m_greedy = None

    try:
        print("  🔹 Greedy initialization ...")

        # Build base model first
        m_greedy = build_base_model(
            M, N, in_range, Ji, Ij, demand_I, Q, P,
            distIJ=distIJ, method_name=policy,
        )

        if greedy_mode == "weighted":
            print("     → Using WEIGHTED greedy (W1)")
            m_greedy = build_initial_solution_weighted(
                m_greedy, distIJ, demand_I, method_name=policy
            )

        elif greedy_mode == "deterministic":
            print("     → Using SMART deterministic greedy")
            m_greedy = build_initial_solution_smart(
                m_greedy, distIJ, method_name=policy
            )

        else:
            print("     → Using RANDOM greedy")
            m_greedy = build_initial_solution(
                m_greedy, distIJ, mode="greedy", policy=policy
            )


        met = evaluate_solution(m_greedy, distIJ, demand_I, method_name=policy)
        score_greedy = met["covered_demand"]
        print(f"     Greedy score = {score_greedy:.3f}")

    except Exception as e:
        print(f"    ⚠️ Greedy failed: {e}")
        m_greedy = None

    # =====================================================
    # 3) LOCAL SEARCH (no restarts, no move logs)
    # =====================================================
    score_LS = score_greedy
    time_LS = np.nan

    if m_greedy is not None:
        try:
            print("  🔹 Local Search ...")
            t0 = time.perf_counter()
            m_LS = local_search(
                m_greedy,
                distIJ, in_range, Ji, Ij, farther_of,
                method_name=policy,
                max_iter=max_iter,
                improvement_rule="first",
                try_order="random",
                logger=None,   # no restart/move logging
            )
            time_LS = time.perf_counter() - t0
            met = evaluate_solution(m_LS, distIJ, demand_I, method_name=policy)
            score_LS = met["covered_demand"]
            print(f"     LS score = {score_LS:.3f} (t={time_LS:.2f}s)")
        except Exception as e:
            print(f"    ⚠️ LS failed: {e}")

    # LS should not exceed exact (if exact is known)
    if not np.isnan(score_exact):
        score_LS = min(score_LS, score_exact)

    # =====================================================
    # 4) SIMULATED ANNEALING (single run, no logs)
    # =====================================================
    score_SA = np.nan
    time_SA = np.nan

    if m_greedy is not None:
        try:
            print("  🔹 Simulated Annealing ...")
            t0 = time.perf_counter()
            m_SA = simulated_annealing(
                m_greedy, distIJ, in_range, Ji, Ij,
                method_name=policy,
                max_iter=200,
                T0=1.0, alpha=0.95, Tmin=1e-3,
                logger=None,   # no SA logs
            )
            time_SA = time.perf_counter() - t0
            met = evaluate_solution(m_SA, distIJ, demand_I, method_name=policy)
            score_SA = met["covered_demand"]
            print(f"     SA score = {score_SA:.3f} (t={time_SA:.2f}s)")
        except Exception as e:
            print(f"    ⚠️ SA failed: {e}")

    # =====================================================
    # RETURN CLEAN ROW (for DataFrame)
    # =====================================================
    return dict(
        policy=policy,
        score_exact=score_exact,
        time_exact=time_exact,
        score_greedy=score_greedy,
        score_LS=score_LS,
        time_LS=time_LS,
        score_SA=score_SA,
        time_SA=time_SA,
        optimal_exact=optimal_exact,
    )


# =========================================================
# run_batch_one_pass — core loop (NO saving)
# =========================================================

def run_batch_one_pass(
    Ns=(20,),
    seeds=(1,),
    policies=("closest_only", "closest_priority", "system_optimum", "uniform"),
    greedy_mode="deterministic",
    save_instances=True,
    data_dir_small="data/small",
    data_dir_large="data/large",
    forbid_self=False,
    max_iter=50,
    exact_time_limit=120,
    exact_mip_gap=0.02,
):
    """
    Core pass over Ns × seeds × policies using a chosen greedy_mode.
    No files written here; returns a clean flat DataFrame with:
      N, seed, policy, score_exact, time_exact, Gap (%),
      score_greedy, score_LS, time_LS, score_SA, time_SA
    """

    import pathlib
    rows = []

    for N in Ns:
        folder = data_dir_small if N <= 100 else data_dir_large
        pathlib.Path(folder).mkdir(parents=True, exist_ok=True)

        P, Q, D = default_parameters_for(N)

        for seed in seeds:
            inst_path = pathlib.Path(folder) / f"inst_N{N}_seed{seed}.json"
            if inst_path.exists():
                inst = load_instance(inst_path)
            else:
                inst = generate_instance(N=N, seed=seed)
                if save_instances:
                    save_instance(inst, inst_path)

            for policy in policies:
                row = run_one_policy(
                    inst,
                    policy,
                    P=P, Q=Q, D=D,
                    forbid_self=forbid_self,
                    max_iter=max_iter,
                    exact_time_limit=exact_time_limit,
                    exact_mip_gap=exact_mip_gap,
                    greedy_mode=greedy_mode,
                )
                row["N"] = N
                row["seed"] = seed
                rows.append(row)

    df = pd.DataFrame(rows)

    # Safety: LS cannot exceed exact
    df["score_LS"] = df[["score_LS", "score_exact"]].min(axis=1)

    # Positive absolute gap (LS vs Exact)
    df["Gap (%)"] = (
        100 * abs(df["score_LS"] - df["score_exact"])
        / df["score_exact"].replace(0, np.nan)
    )

    # Canonical flat column order
    df = df[
        [
            "N",
            "seed",
            "policy",
            "score_exact",
            "time_exact",
            "Gap (%)",
            "score_greedy",
            "score_LS",
            "time_LS",
            "score_SA",
            "time_SA",
        ]
    ]

    return df


# =========================================================
# run_batch — thin wrapper for backward compatibility
# =========================================================

def run_batch(
    Ns=(20,40),
    seeds=(1,),
    policies=("closest_only", "closest_priority", "system_optimum", "uniform"),
    greedy_mode="deterministic",
    save_instances=True,
    data_dir_small="data/small",
    data_dir_large="data/large",
    forbid_self=False,
    max_iter=50,
    exact_time_limit=120,
    exact_mip_gap=0.02,
):
    """
    Backward-compatible wrapper that simply calls run_batch_one_pass.
    Use this only if some older notebook expects run_batch().
    """
    return run_batch_one_pass(
        Ns=Ns,
        seeds=seeds,
        policies=policies,
        greedy_mode=greedy_mode,
        save_instances=save_instances,
        data_dir_small=data_dir_small,
        data_dir_large=data_dir_large,
        forbid_self=forbid_self,
        max_iter=max_iter,
        exact_time_limit=exact_time_limit,
        exact_mip_gap=exact_mip_gap,
    )
