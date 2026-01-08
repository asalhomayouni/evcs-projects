import numpy as np
import pandas as pd
import time
from pathlib import Path
from pyomo.opt import TerminationCondition

from evcs.geom import build_arcs
from evcs.model import build_base_model
from evcs.methods import (
    build_initial_solution_weighted,
    reconstruction_greedy,
    local_search,
    evaluate_solution,
    compute_farther,
    destroy_partial,
    compare_solutions,
)
from evcs.solve import solve_model

from scripts.randomInstance import generate_instance, save_instance, load_instance


class DRLogger:
    def __init__(self):
        self.it = []
        self.score = []
        self.best = []
        self.time = []
        self.k_remove = []
        self.mode = []

    def log(self, it, score, best, elapsed, k_remove=None, mode=None):
        self.it.append(it)
        self.score.append(score)
        self.best.append(best)
        self.time.append(elapsed)
        self.k_remove.append(k_remove)
        self.mode.append(mode)

    def to_df(self):
        return pd.DataFrame({
            "iteration": self.it,
            "score": self.score,
            "best": self.best,
            "time": self.time,
            "k_remove": self.k_remove,
            "destroy_mode": self.mode,
        })

def default_parameters_binary(N):
    P = max(2, int(round((N ** 0.5) / 1.2)))
    avg_demand = 3.0
    Q = (N * avg_demand) / max(1, P)
    D = 3.5 if N <= 60 else 2.0
    return P, Q, D

def default_parameters_integer(N):
    P_sites, Q, _ = default_parameters_binary(N)
    P_chargers = int(1.2 * P_sites)
    D = 1.5
    return P_chargers, Q, D

def run_one_policy(
    inst,
    policy,
    P,
    Q,
    D,
    forbid_self: bool = False,
    max_iter: int = 50,
    dr_time_limit: float = 300.0,
    dr_log_every: int = 1,
    exact_time_limit: float = 400,
    exact_mip_gap: float = 0.01,
    greedy_mode: str = "deterministic",
    destroy_mode: str = "area",
    allow_multi_charger: bool = False,
    max_chargers_per_site: int | None = None,
    seed: int | None = None,
):
    """
    One run: exact baseline -> greedy init -> LS -> DR loop (time-budgeted).
    Works for binary stations and integer chargers (single-period).
    """
    if seed is not None:
        np.random.seed(seed)

    # Normalize greedy mode names
    if greedy_mode in ("W1", "weighted_W1"):
        greedy_init_mode = "W1"
        greedy_recon_mode = "weighted_W1"
    elif greedy_mode in ("W2", "weighted_W2"):
        greedy_init_mode = "W2"
        greedy_recon_mode = "weighted_W2"
    elif greedy_mode == "deterministic":
        greedy_init_mode = "W1"  # for integer chargers, deterministic here is okay but W1 is a strong default
        greedy_recon_mode = "deterministic"
    else:
        raise ValueError(f"Unknown greedy_mode: {greedy_mode}")

    coords_I, coords_J = inst["coords_I"], inst["coords_J"]
    I_idx = inst.get("I_idx", list(range(len(coords_I))))
    J_idx = inst.get("J_idx", list(range(len(coords_J))))
    demand_I = inst["demand_I"]
    M, N = len(I_idx), len(J_idx)

    distIJ, in_range, Ji, Ij = build_arcs(
        coords_I, coords_J, D=D, forbid_self=forbid_self,
        I_idx=I_idx, J_idx=J_idx
    )

    # FIX: compute_farther expects Ji (i -> reachable j list)
    farther_of = compute_farther(distIJ, in_range, Ji)

    # =================================================
    # 1) EXACT baseline
    # =================================================
    score_exact = np.nan
    time_exact = np.nan
    optimal_exact = False
    m_exact = None
    try:
        m_exact = build_base_model(
            M, N, in_range, Ji, Ij, demand_I, Q, P,
            distIJ=distIJ, method_name=policy,
            allow_multi_charger=allow_multi_charger,
            max_chargers_per_site=max_chargers_per_site,
        )
        t0 = time.perf_counter()
        res = solve_model(
            m_exact, verbose=False,
            time_limit=exact_time_limit,
            mip_gap=(exact_time_limit and exact_mip_gap)
        )
        time_exact = time.perf_counter() - t0
        score_exact = evaluate_solution(m_exact, distIJ, demand_I, method_name=policy)["covered_demand"]
        optimal_exact = (getattr(res, "termination_condition", None) == TerminationCondition.optimal)
    except Exception as e:
        print(f"⚠️ Exact solve failed/skipped: {e}")

    # =================================================
    # 2) GREEDY init
    # =================================================
    m0 = build_base_model(
        M, N, in_range, Ji, Ij, demand_I, Q, P,
        distIJ=distIJ, method_name=policy,
        allow_multi_charger=allow_multi_charger,
        max_chargers_per_site=max_chargers_per_site,
    )

    try:
        m_greedy = build_initial_solution_weighted(
            m0, distIJ, demand_I,
            method_name=policy,
            weight_mode=greedy_init_mode
        )
    except Exception as e:
        print(f"⚠️ Greedy init failed, using base model: {e}")
        m_greedy = m0

    score_greedy = evaluate_solution(m_greedy, distIJ, demand_I, method_name=policy)["covered_demand"]

    # =================================================
    # 3) LOCAL SEARCH
    # =================================================
    m_LS = m_greedy
    time_LS = np.nan
    try:
        t0 = time.perf_counter()
        m_LS = local_search(
            m_greedy,
            distIJ, in_range, Ji, Ij, farther_of,
            method_name=policy,
            max_iter=max_iter,
            improvement_rule="first",
            try_order="random",
            logger=None
        )
        time_LS = time.perf_counter() - t0
    except Exception as e:
        print(f"⚠️ LS failed, using greedy: {e}")
        m_LS = m_greedy

    score_LS = evaluate_solution(m_LS, distIJ, demand_I, method_name=policy)["covered_demand"]

    # =================================================
    # 4) DR init
    # =================================================
    m_best = m_LS
    best_score = score_LS

    logger_dr = DRLogger()
    start_DR = time.perf_counter()
    it = 0

    # =================================================
    # 5) DR loop (time-budgeted)
    # =================================================
    while (time.perf_counter() - start_DR) < dr_time_limit:
        it += 1

        dm = (destroy_mode or "random").lower()
        if dm in ("area", "cluster"):
            ratio = np.random.uniform(0.3, 0.7)
        elif dm in ("demand_low", "demand_high"):
            ratio = np.random.uniform(0.4, 0.8)
        else:
            ratio = np.random.uniform(0.1, 0.9)

        # Important:
        #  - binary: k_remove is number of stations removed (<= P sites)
        #  - integer: k_remove is number of chargers removed (<= total P chargers)
        k_remove = max(1, int(ratio * P))

        # clone best before mutation
        m_ref = m_best.clone()

        # destroy
        m_tmp = destroy_partial(
            m_ref,
            k_remove=k_remove,
            mode=destroy_mode,
            coords_J=coords_J,
            demand_I=demand_I,
            radius=None,
            seed=None
        )

        # reconstruct
        m_tmp = reconstruction_greedy(
            m_tmp, distIJ, demand_I, D,
            method_name=policy, greedy_mode=greedy_recon_mode
        )

        # local search
        m_tmp = local_search(
            m_tmp,
            distIJ, in_range, Ji, Ij, farther_of,
            method_name=policy,
            max_iter=max_iter,
            improvement_rule="first",
            try_order="random",
            logger=None
        )

        # evaluate
        new_score = evaluate_solution(m_tmp, distIJ, demand_I, method_name=policy)["covered_demand"]

        # accept if improved
        if new_score > best_score + 1e-6:
            best_score = new_score
            m_best = m_tmp

        if (it % max(1, dr_log_every)) == 0:
            logger_dr.log(it, new_score, best_score, time.perf_counter() - start_DR, k_remove=k_remove, mode=destroy_mode)

    # =================================================
    # 6) Side-by-side compare info
    # =================================================
    cmp_exact_best = None
    if m_exact is not None:
        try:
            cmp_exact_best = compare_solutions(m_exact, m_best, demand_I)
        except Exception:
            cmp_exact_best = None

    return dict(
        policy=policy,
        score_exact=score_exact,
        time_exact=time_exact,
        optimal_exact=optimal_exact,
        score_greedy=score_greedy,
        score_LS=score_LS,
        time_LS=time_LS,
        score_DR=best_score,
        DR_log=logger_dr.to_df(),
        compare_exact_vs_best=cmp_exact_best,
        m_exact=m_exact,
        m_best=m_best,
    )

def run_one_policy_multi(
    inst,
    policy: str,
    P_T,                 # list length T: installs per period (e.g., [2,1,1,0])
    Q: float,
    D: float,
    T: int | None = None,
    forbid_self: bool = False,
    exact_time_limit: float = 400,
    exact_mip_gap: float = 0.01,
    greedy_mode: str = "score",   # currently one simple greedy strategy
    max_chargers_per_site: int | None = None,
    cumulative_install: bool = True,
    seed: int | None = None,
):
    """
    Multi-period version:
      1) EXACT baseline (optional)
      2) GREEDY install schedule baseline + greedy assignment

    IMPORTANT:
      - This function is additive; it won't affect single-period.
      - It expects inst["demand_IT"] = demand per period (shape [T][M]) OR dict {(i,t): val}.
      - P_T is the per-period install budget: sum_j u[j,t] <= P_T[t].
    """
    # Local imports to avoid breaking single-period if you haven't added multi functions yet
    from evcs.geom import build_arcs
    from evcs.solve import solve_model
    from evcs.model import build_multi_period_model
    from evcs.methods import reassign_y_greedy_multi, evaluate_solution_multi

    import numpy as np
    import time
    from pyomo.opt import TerminationCondition

    if seed is not None:
        np.random.seed(seed)

    coords_I, coords_J = inst["coords_I"], inst["coords_J"]
    I_idx = inst.get("I_idx", list(range(len(coords_I))))
    J_idx = inst.get("J_idx", list(range(len(coords_J))))

    demand_IT = inst.get("demand_IT", None)
    if demand_IT is None:
        raise KeyError('inst must contain "demand_IT" for multi-period runs.')

    # Determine T
    if T is None:
        T = len(demand_IT) if not isinstance(demand_IT, dict) else (max(t for (_, t) in demand_IT.keys()) + 1)

    if len(P_T) != T:
        raise ValueError(f"P_T length must be T={T}, got {len(P_T)}")

    M, N = len(I_idx), len(J_idx)

    distIJ, in_range, Ji, Ij = build_arcs(
        coords_I, coords_J, D=D, forbid_self=forbid_self,
        I_idx=I_idx, J_idx=J_idx
    )

    # =================================================
    # 1) EXACT baseline (MIP)
    # =================================================
    score_exact = np.nan
    time_exact = np.nan
    optimal_exact = False
    m_exact = None

    try:
        m_exact = build_multi_period_model(
            M=M, N=N, T=T,
            in_range=in_range, Ji=Ji, Ij=Ij,
            demand_IT=demand_IT,
            Q=Q, P_T=P_T,
            distIJ=distIJ,
            method_name=policy,
            allow_multi_charger=True,
            max_chargers_per_site=max_chargers_per_site,
            cumulative_install=cumulative_install,
        )
        t0 = time.perf_counter()
        res = solve_model(
            m_exact,
            verbose=False,
            time_limit=exact_time_limit,
            mip_gap=(exact_time_limit and exact_mip_gap),
        )
        time_exact = time.perf_counter() - t0

        score_exact = evaluate_solution_multi(m_exact, demand_IT)["covered_demand"]
        optimal_exact = (getattr(res, "termination_condition", None) == TerminationCondition.optimal)
    except Exception as e:
        print(f"⚠️ Exact multi-period solve failed/skipped: {e}")

    # =================================================
    # 2) GREEDY install schedule baseline
    #    (simple: place installs at sites with highest weighted reachable demand each period)
    # =================================================
    m0 = build_multi_period_model(
        M=M, N=N, T=T,
        in_range=in_range, Ji=Ji, Ij=Ij,
        demand_IT=demand_IT,
        Q=Q, P_T=P_T,
        distIJ=distIJ,
        method_name=policy,
        allow_multi_charger=True,
        max_chargers_per_site=max_chargers_per_site,
        cumulative_install=cumulative_install,
    )

    # Helper to fetch demand(i,t)
    def dem(i, t):
        if isinstance(demand_IT, dict):
            return float(demand_IT[(i, t)])
        return float(demand_IT[t][i])

    # 2.1 initialize installs u=0
    for j in m0.J:
        for t in m0.T:
            if hasattr(m0, "u"):
                m0.u[j, t].value = 0

    # 2.2 greedy installs per period
    eps = 1e-6
    U = int(m0.U.value) if hasattr(m0, "U") else 0

    for t in range(T):
        # score each site j by "reachable demand / distance"
        scores = {}
        for j in range(N):
            s = 0.0
            for i in Ij.get(j, []):
                dij = distIJ[i][j]
                s += dem(i, t) / (dij + eps)
            scores[j] = s

        # allocate P_T[t] installs to best sites (allow multiple at same site)
        for _ in range(int(P_T[t])):
            j_best = max(scores, key=scores.get)
            # respect U bound
            if U > 0 and (m0.u[j_best, t].value or 0) >= U:
                scores[j_best] = -1e30
                continue
            m0.u[j_best, t].value = int(m0.u[j_best, t].value or 0) + 1

    # 2.3 build x and z consistent with dynamics
    for j in range(N):
        cum = 0
        for t in range(T):
            cum = cum + int(m0.u[j, t].value or 0) if cumulative_install else int(m0.u[j, t].value or 0)
            m0.x[j, t].value = cum
            m0.z[j, t].value = 1 if cum > 0 else 0

    # 2.4 greedy assignment per period
    try:
        m_greedy = reassign_y_greedy_multi(m0, distIJ, Ji, policy)
    except Exception as e:
        print(f"⚠️ Greedy assignment failed: {e}")
        m_greedy = m0

    metrics_greedy = evaluate_solution_multi(m_greedy, demand_IT)
    score_greedy = metrics_greedy["covered_demand"]

    return dict(
        policy=policy,
        T=T,
        P_T=list(P_T),
        score_exact=score_exact,
        time_exact=time_exact,
        optimal_exact=optimal_exact,
        score_greedy=score_greedy,
        m_exact=m_exact,
        m_best=m_greedy,   # for now, greedy is the "best" multi-period heuristic baseline
    )
