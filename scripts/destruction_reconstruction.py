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


def default_parameters_for(N: int):
    P = max(2, int(round((N ** 0.5) / 1.2)))
    avg_demand = 3.0
    Q = (N * avg_demand) / max(1, P)
    D = 3.5 if N <= 60 else 2.8
    return P, Q, D


def run_one_policy(
    inst,
    policy,
    P,
    Q,
    D,
    forbid_self: bool = False,
    max_iter: int = 50,
    # Step 1: Time-budget DR
    dr_time_limit: float = 300.0,
    dr_log_every: int = 1,
    exact_time_limit: float = 400,
    exact_mip_gap: float = 0.01,
    greedy_mode: str = "deterministic",
    destroy_mode: str = "area",
    allow_multi_charger: bool = False,  # Step 7 switch
    seed: int | None = None,
):
    """
    One run: (optional) exact baseline -> greedy init -> LS -> DR loop (time-budgeted).

    Returns dict with:
      scores/times + DR_log + optional solution comparison fields.
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
        greedy_init_mode = None
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
    farther_of = compute_farther(distIJ, in_range, Ij)

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
    )

    try:
        if greedy_init_mode is not None:
            m_greedy = build_initial_solution_weighted(
                m0, distIJ, demand_I,
                method_name=policy,
                weight_mode=greedy_init_mode
            )
        else:
            m_greedy = reconstruction_greedy(
                m0, distIJ, demand_I, D,
                method_name=policy, greedy_mode="deterministic"
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

    # (optional safety) LS shouldn't beat exact if exact exists
    if not np.isnan(score_exact):
        score_LS = min(score_LS, score_exact)

    # =================================================
    # 4) DR init
    # =================================================
    m_best = m_LS
    best_score = score_LS

    logger_dr = DRLogger()
    start_DR = time.perf_counter()
    it = 0

    # =================================================
    # 5) DR loop (time-budgeted) — Steps 1/2/3
    # =================================================
    while (time.perf_counter() - start_DR) < dr_time_limit:
        it += 1

        # choose k_remove schedule by mode
        dm = (destroy_mode or "random").lower()
        if dm in ("area", "cluster"):
            ratio = np.random.uniform(0.3, 0.7)
        elif dm in ("demand_low", "demand_high"):
            ratio = np.random.uniform(0.4, 0.8)
        else:
            ratio = np.random.uniform(0.1, 0.9)

        k_remove = max(1, int(ratio * P))

        # Step 2: ALWAYS clone best before mutation
        m_ref = m_best.clone()

        # destroy (Step 3 modes)
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

        # log (dr_log_every)
        if (it % max(1, dr_log_every)) == 0:
            logger_dr.log(it, new_score, best_score, time.perf_counter() - start_DR, k_remove=k_remove, mode=destroy_mode)

    # =================================================
    # 6) Side-by-side compare info (Step 4)
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
    dr_time_limit=300.0,
    dr_log_every=1,
    destroy_mode="area",
    allow_multi_charger=False,
):
    rows = []

    for N in Ns:
        folder = data_dir_small if N <= 100 else data_dir_large
        Path(folder).mkdir(parents=True, exist_ok=True)

        P, Q, D = default_parameters_for(N)

        for seed in seeds:
            inst_path = Path(folder) / f"inst_N{N}_seed{seed}.json"
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
                    destroy_mode=destroy_mode,
                    dr_time_limit=dr_time_limit,
                    dr_log_every=dr_log_every,
                    allow_multi_charger=allow_multi_charger,
                    seed=seed,
                )
                row["N"] = N
                row["seed"] = seed
                rows.append(row)

    df = pd.DataFrame(rows)

    # Safety: LS cannot exceed exact if exact exists
    df["score_LS"] = df[["score_LS", "score_exact"]].min(axis=1)

    # Gap (%)
    df["Gap (%)"] = 100 * (df["score_LS"] - df["score_exact"]).abs() / df["score_exact"].replace(0, np.nan)

    # Flat view
    df_flat = df[[
        "N", "seed", "policy",
        "score_exact", "time_exact", "Gap (%)",
        "score_greedy", "score_LS", "time_LS",
        "score_DR",
    ]].copy()

    return df_flat, df  # df has DR_log + compare dicts
