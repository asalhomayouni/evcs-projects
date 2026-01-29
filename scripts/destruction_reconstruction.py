import numpy as np
import pandas as pd
import time
from pathlib import Path
from pyomo.opt import TerminationCondition

from evcs.geom import build_arcs
from evcs.model import build_base_model ,build_multi_period_model
from evcs.methods import (
    build_initial_solution_weighted,
    reconstruction_greedy,
    local_search,
    evaluate_solution,
    compute_farther,
    destroy_partial,
    compare_solutions,
    apply_method_multi,
    reassign_y_greedy_multi,
    evaluate_solution_multi,
    sync_solution_state,
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
        self.accepted = []
        self.unique = []

    def log(self, it, score, best, elapsed, k_remove=None, mode=None, accepted=None, unique=None):
        self.it.append(it)
        self.score.append(score)
        self.best.append(best)
        self.time.append(elapsed)
        self.k_remove.append(k_remove)
        self.mode.append(mode)
        self.accepted.append(accepted)
        self.unique.append(unique)

    def to_df(self):
        return pd.DataFrame({
            "iteration": self.it,
            "score": self.score,
            "best": self.best,
            "time": self.time,
            "k_remove": self.k_remove,
            "destroy_mode": self.mode,
            "accepted": self.accepted,
            "unique": self.unique,
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
    P_T,
    Q: float,
    D: float,
    T: int,
    forbid_self: bool = False,
    exact_time_limit: float = 120,
    exact_mip_gap: float = 0.10,
    max_chargers_per_site: int | None = None,
    cumulative_install: bool = True,
    seed: int | None = None,
    verbose: bool = False,
):
    import time
    import numpy as np
    from pyomo.opt import TerminationCondition

    from evcs.model import build_multi_period_model
    from evcs.methods import (
        compute_farther,
        apply_method_multi,
        sync_solution_state,
        reassign_y_greedy_multi,
        evaluate_solution_multi,
    )

    coords_I, coords_J = inst["coords_I"], inst["coords_J"]
    demand_IT = inst["demand_IT"]  # list length T, each length M
    M = len(coords_I)
    N = len(coords_J)

    # arcs
    distIJ, in_range, Ji, Ij = build_arcs(
        coords_I, coords_J, D=D, forbid_self=forbid_self
    )

    def _extract_mip_gap_and_bound(res, sense: str = "max"):
        gap = None
        best_bound = None
        incumbent = None

        try:
            gap = getattr(res.solver, "mip_gap", None)
            if gap is None:
                gap = getattr(res.solver, "gap", None)
        except Exception:
            pass

        sm = None
        try:
            sm = getattr(res.solver, "_solver_model", None)
        except Exception:
            sm = None

        if sm is not None:
            try:
                if hasattr(sm, "MIPGap"):
                    gap = float(sm.MIPGap)
                if hasattr(sm, "ObjBound"):
                    best_bound = float(sm.ObjBound)
                if hasattr(sm, "ObjVal"):
                    incumbent = float(sm.ObjVal)
            except Exception:
                pass

            try:
                if gap is None and hasattr(sm, "solution") and hasattr(sm.solution, "MIP"):
                    gap = float(sm.solution.MIP.get_mip_relative_gap())
                if best_bound is None and hasattr(sm, "solution") and hasattr(sm.solution, "MIP"):
                    best_bound = float(sm.solution.MIP.get_best_objective())
            except Exception:
                pass

        if gap is None and (best_bound is not None) and (incumbent is not None):
            denom = max(1.0, abs(incumbent))
            if sense.lower().startswith("max"):
                gap = max(0.0, (best_bound - incumbent) / denom)
            else:
                gap = max(0.0, (incumbent - best_bound) / denom)

        return gap, best_bound, incumbent

    # -------------------------
    # EXACT (time-limited)
    # -------------------------
    score_exact = None
    time_exact = None
    m_exact = None
    exact_termination = None
    proven_optimal_exact = None
    exact_gap = None
    exact_bound = None
    exact_incumbent_obj = None

    try:
        m_exact = build_multi_period_model(
            M=M, N=N, T=T,
            in_range=in_range, Ji=Ji, Ij=Ij,
            demand_IT=demand_IT, Q=Q, P_T=P_T,
            distIJ=distIJ,
            method_name=policy,
            max_chargers_per_site=max_chargers_per_site,
            cumulative_install=cumulative_install,
        )

        farther_of = compute_farther(distIJ, in_range, Ji)
        m_exact = apply_method_multi(m_exact, policy, distIJ, in_range, Ji, Ij, farther_of, verbose=False)

        t0 = time.perf_counter()
        res = solve_model(
            m_exact,
            verbose=verbose,
            time_limit=exact_time_limit,
            mip_gap=exact_mip_gap
        )
        time_exact = time.perf_counter() - t0

        tc = None
        try:
            tc = res.solver.termination_condition
        except Exception:
            tc = getattr(res, "termination_condition", None)

        exact_termination = tc
        proven_optimal_exact = (tc == TerminationCondition.optimal)
        exact_gap, exact_bound, exact_incumbent_obj = _extract_mip_gap_and_bound(res, sense="max")
        score_exact = float(evaluate_solution_multi(m_exact, demand_IT)["covered_demand"])

    except Exception as e:
        if verbose:
            print("Exact failed:", e)

    # -------------------------
    # GREEDY schedule + greedy assign
    # -------------------------
    m_g = build_multi_period_model(
        M=M, N=N, T=T,
        in_range=in_range, Ji=Ji, Ij=Ij,
        demand_IT=demand_IT, Q=Q, P_T=P_T,
        distIJ=distIJ,
        method_name=policy,
        max_chargers_per_site=max_chargers_per_site,
        cumulative_install=cumulative_install,
    )

    farther_of = compute_farther(distIJ, in_range, Ji)
    m_g = apply_method_multi(m_g, policy, distIJ, in_range, Ji, Ij, farther_of, verbose=False)

    # clear state
    for t in m_g.T:
        for j in m_g.J:
            m_g.u[j, t].value = 0
            m_g.x[j, t].value = 0
            m_g.z[j, t].value = 0

    # U cap (prefer model’s U)
    U = int(m_g.U.value) if hasattr(m_g, "U") else (
        int(max_chargers_per_site) if max_chargers_per_site is not None else int(sum(P_T))
    )

    # adjacency j -> list(i)
    Ij_int = {}
    for (i, j) in in_range:
        Ij_int.setdefault(int(j), []).append(int(i))

    def x_now(j, t_int):
        if cumulative_install:
            return sum(int(m_g.u[j, tt].value or 0) for tt in m_g.T if int(tt) <= int(t_int))
        return int(m_g.u[j, t_int].value or 0)

    t0 = time.perf_counter()

    # =========================
    # STRONG GREEDY SCHEDULING (marginal uncovered demand, capacity-aware update)
    # =========================
    for t in range(T):
        uncovered = set(range(len(demand_IT[t])))

        for _ in range(int(P_T[t])):
            scored = []
            for j in m_g.J:
                if x_now(j, t) >= U:
                    continue

                score = sum(
                    float(demand_IT[t][i])
                    for i in Ij_int.get(int(j), [])
                    if int(i) in uncovered
                )
                scored.append((score, int(j)))

            if not scored:
                break

            scored.sort(reverse=True, key=lambda x: x[0])
            best_score, best_j = scored[0]

            if best_score <= 1e-12:
                break

            m_g.u[best_j, t].value = int(m_g.u[best_j, t].value or 0) + 1

            # capacity-aware uncovered update (remove only what one charger can serve)
            cap = Q
            reach_nodes = [int(i) for i in Ij_int.get(int(best_j), []) if int(i) in uncovered]
            reach_nodes.sort(key=lambda i: float(demand_IT[t][i]), reverse=True)

            used = 0.0
            for i in reach_nodes:
                di = float(demand_IT[t][i])
                if used + di <= cap + 1e-9:
                    uncovered.discard(i)
                    used += di

    # sync x/z from u then assign y
    sync_solution_state(m_g, cumulative_install=cumulative_install)
    m_g = reassign_y_greedy_multi(m_g, distIJ, Ji, method_name=policy, cumulative_install=cumulative_install)

    score_greedy = float(evaluate_solution_multi(m_g, demand_IT)["covered_demand"])
    time_greedy = time.perf_counter() - t0

    return dict(
        policy=policy,
        T=T,
        P_T=list(P_T),
        score_exact=score_exact,
        time_exact=time_exact,
        exact_termination=exact_termination,
        proven_optimal_exact=proven_optimal_exact,
        score_greedy=score_greedy,
        time_greedy=time_greedy,
        m_exact=m_exact,
        m_best=m_g,
        distIJ=distIJ,
        Ji=Ji,
        Ij=Ij,
        in_range=in_range,
        exact_gap=exact_gap,
        exact_bound=exact_bound,
        exact_incumbent_obj=exact_incumbent_obj,
    )

    # sync x/z from u then assign y
    sync_solution_state(m_g, cumulative_install=cumulative_install)
    m_g = reassign_y_greedy_multi(m_g, distIJ, Ji, method_name=policy, cumulative_install=cumulative_install)

    score_greedy = float(evaluate_solution_multi(m_g, demand_IT)["covered_demand"])
    time_greedy = time.perf_counter() - t0

    return dict(
        policy=policy,
        T=T,
        P_T=list(P_T),
        score_exact=score_exact,
        time_exact=time_exact,
        exact_termination=exact_termination,
        proven_optimal_exact=proven_optimal_exact,
        score_greedy=score_greedy,
        time_greedy=time_greedy,
        m_exact=m_exact,
        m_best=m_g,
        distIJ=distIJ,
        Ji=Ji,
        Ij=Ij,
        in_range=in_range,
        exact_gap=exact_gap,
        exact_bound=exact_bound,
        exact_incumbent_obj=exact_incumbent_obj,
    )


# =========================================================
# Priority-2: Multi-period DR (Destroy / Reconstruct / Loop)
# =========================================================

def _clone_u_matrix(m):
    """Return u as a plain dict {(j,t): int} for safe copying."""
    U = {}
    for j in m.J:
        for t in m.T:
            U[(int(j), int(t))] = int(m.u[j, t].value or 0)
    return U


def _apply_u_matrix(m, Udict):
    Q = float(m.Q.value)

    """Write dict {(j,t): int} back into m.u[j,t]."""
    for j in m.J:
        jj = int(j)
        for t in m.T:
            tt = int(t)
            m.u[j, t].value = int(Udict.get((jj, tt), 0))


import numpy as np

def destroy_multi_u(
    Udict,
    P_T,
    frac_remove: float = 0.20,
    mode: str = "k_units",
    seed: int | None = None,
    site_cap: int | None = None,          # NEW
    cumulative_install: bool = True,      # NEW
):

    if seed is not None:
        np.random.seed(seed)

    mode = (mode or "k_units").lower()

    # IMPORTANT: work on a COPY (do NOT mutate input)
    U_new = dict(Udict)

    keys = list(U_new.keys())
    if not keys:
        return U_new, 0

    total = sum(int(v) for v in U_new.values())
    if total <= 0:
        return U_new, 0

    # infer sets
    Js = sorted({int(j) for (j, t) in U_new.keys()})
    Ts = sorted({int(t) for (j, t) in U_new.keys()})

    # ---------- site_all ----------
    if mode in ("site_all", "site"):
        candidates = []
        for j in Js:
            totj = sum(int(U_new[(j, t)]) for t in Ts)
            if totj > 0:
                candidates.append(j)

        if not candidates:
            return U_new, 0

        j_star = int(np.random.choice(candidates))
        before = sum(int(U_new[(j_star, t)]) for t in Ts)
        for t in Ts:
            U_new[(j_star, t)] = 0
        return U_new, before

    # ---------- site_future ----------
    if mode in ("site_future", "future"):
        candidates = []
        for j in Js:
            totj = sum(int(U_new[(j, t)]) for t in Ts)
            if totj > 0:
                candidates.append(j)

        if not candidates:
            return U_new, 0

        j_star = int(np.random.choice(candidates))
        t0 = int(np.random.choice(Ts))

        before = sum(int(U_new[(j_star, t)]) for t in Ts if t >= t0)
        if before <= 0:
            return U_new, 0

        for t in Ts:
            if t >= t0:
                U_new[(j_star, t)] = 0
        return U_new, before

    # ---------- site_swap ----------
    if mode in ("site_swap", "swap"):
        # totals per site
        tot_by_j = {j: sum(int(U_new[(j, t)]) for t in Ts) for j in Js}

        open_sites = [j for j, totj in tot_by_j.items() if totj > 0]
        if not open_sites:
            return U_new, 0

        closed_sites = [j for j, totj in tot_by_j.items() if totj == 0]

        j_out = int(np.random.choice(open_sites))

        # choose j_in != j_out, prefer closed site
        if closed_sites:
            closed_no_out = [j for j in closed_sites if j != j_out]
            if closed_no_out:
                j_in = int(np.random.choice(closed_no_out))
            else:
                # all closed sites equals j_out (rare); fallback
                cand = [j for j in Js if j != j_out]
                if not cand:
                    return U_new, 0
                j_in = int(np.random.choice(cand))
        else:
            cand = [j for j in Js if j != j_out]
            if not cand:
                return U_new, 0
            j_in = int(np.random.choice(cand))

        before = sum(int(U_new[(j_out, t)]) for t in Ts)
        if before <= 0:
            return U_new, 0

        # If no cap enforcement requested, move everything period-by-period
        if site_cap is None:
            for t in Ts:
                v = int(U_new[(j_out, t)])
                if v > 0:
                    U_new[(j_in, t)] = int(U_new[(j_in, t)]) + v
                    U_new[(j_out, t)] = 0
            return U_new, before

        # With cap enforcement
        Ts_sorted = sorted(Ts)

        if cumulative_install:
            # enforce cumulative capacity at j_in across periods
            cum_in = 0
            for t in Ts_sorted:
                cum_in += int(U_new[(j_in, t)])
                v = int(U_new[(j_out, t)])
                if v <= 0:
                    U_new[(j_out, t)] = 0
                    continue

                remaining = max(0, int(site_cap) - cum_in)
                add = min(v, remaining)
                if add > 0:
                    U_new[(j_in, t)] = int(U_new[(j_in, t)]) + add
                    cum_in += add

                # remove from j_out regardless (this is a destroy operator)
                U_new[(j_out, t)] = 0

            return U_new, before

        # not cumulative: cap per period
        for t in Ts_sorted:
            v = int(U_new[(j_out, t)])
            if v <= 0:
                U_new[(j_out, t)] = 0
                continue

            remaining = max(0, int(site_cap) - int(U_new[(j_in, t)]))
            add = min(v, remaining)
            if add > 0:
                U_new[(j_in, t)] = int(U_new[(j_in, t)]) + add

            U_new[(j_out, t)] = 0

        return U_new, before

    # ---------- unit-based destroys ----------
    if mode in ("k_units", "k"):
        k_remove = max(1, int(np.ceil(frac_remove * total)))
    else:
        # "random_unit" (old weak behavior) or any unknown mode -> fallback
        k_remove = max(1, int(np.ceil(frac_remove * total)))

    k_remove = min(k_remove, total)

    donors = [k for k, v in U_new.items() if int(v) > 0]
    if not donors:
        return U_new, 0

    weights = np.array([max(1e-12, float(U_new[k])) for k in donors], dtype=float)
    weights /= weights.sum()

    removed = 0
    for _ in range(k_remove):
        idx = int(np.random.choice(len(donors), p=weights))
        key = donors[idx]
        if U_new[key] > 0:
            U_new[key] -= 1
            removed += 1

    return U_new, removed



def reconstruct_multi_u_greedy(
    m_template,
    Udict_partial,
    distIJ,
    demand_IT,
    P_T,
    policy: str,
    cumulative_install: bool = True,
):
    """
    STRONG deterministic reconstruction (marginal uncovered demand):
      - Start from partial u
      - Fill missing installs period-by-period using marginal uncovered demand scoring
      - Capacity-aware uncovered update (remove only nodes that can fit into 1 charger capacity)
      - Sync x/z and then assign y using strong reassign_y_greedy_multi
    """
    # Clone the model template (safe) so we don't mutate external object
    m = m_template.clone()

    # 1) set partial u
    _apply_u_matrix(m, Udict_partial)

    # Cache Q once (IMPORTANT)
    Q = float(m.Q.value)

    # 2) fill missing per period
    T = len(P_T)

    # build Ij_int from arcs for reach scoring
    Ij_int = {}
    for (i, j) in m.Arcs:
        Ij_int.setdefault(int(j), []).append(int(i))

    # per-site cap
    U_cap = int(m.U.value) if hasattr(m, "U") else int(max(P_T))

    def x_now(j, t):
        # chargers at (j,t) implied by u
        if cumulative_install:
            return sum(int(m.u[j, tt].value or 0) for tt in m.T if int(tt) <= int(t))
        return int(m.u[j, t].value or 0)

    for t in range(T):
        already = sum(int(m.u[j, t].value or 0) for j in m.J)
        missing = max(0, int(P_T[t]) - int(already))
        if missing <= 0:
            continue

        # track uncovered demand nodes in this period
        uncovered = set(range(len(demand_IT[t])))

        for _ in range(missing):
            cands = []
            for j in m.J:
                if x_now(j, t) >= U_cap:
                    continue

                # marginal gain = reachable demand among *uncovered* nodes
                score = sum(
                    float(demand_IT[t][i])
                    for i in Ij_int.get(int(j), [])
                    if int(i) in uncovered
                )
                cands.append((score, int(j)))

            if not cands:
                break

            cands.sort(reverse=True, key=lambda x: x[0])
            best_score, best_j = cands[0]

            # If no marginal gain remains, stop placing chargers this period
            if best_score <= 1e-12:
                break

            # install one charger
            m.u[best_j, t].value = int(m.u[best_j, t].value or 0) + 1

            # capacity-aware uncovered update: remove only nodes that fit into 1 charger capacity
            cap = Q
            reach_nodes = [
                int(i) for i in Ij_int.get(int(best_j), [])
                if int(i) in uncovered
            ]
            reach_nodes.sort(key=lambda i: float(demand_IT[t][i]), reverse=True)

            used = 0.0
            for i in reach_nodes:
                di = float(demand_IT[t][i])
                if used + di <= cap + 1e-9:
                    uncovered.discard(i)
                    used += di

    # 3) sync x,z then assign y
    sync_solution_state(m, cumulative_install=cumulative_install)
    m = reassign_y_greedy_multi(
        m, distIJ, Ji=None, method_name=policy, cumulative_install=cumulative_install
    )

    return m


def run_DR_multi(
    inst,
    policy: str,
    P_T,
    Q: float,
    D: float,
    T: int,
    max_iter: int = 200,
    dr_time_limit: float = 120.0,
    frac_remove: float = 0.20,
    destroy_mode: str = "random",   # used only if adaptive_destroy=False
    exact_time_limit: float = 120,
    exact_mip_gap: float = 0.10,
    max_chargers_per_site: int | None = None,
    cumulative_install: bool = True,
    seed: int | None = None,
    verbose: bool = False,
    accept_epsilon: float = 0.02,

    # --- adaptive destroy knobs ---
    adaptive_destroy: bool = True,
    destroy_modes: tuple = ("site_all", "site_future", "site_swap", "k_units"),
    update_every: int = 25,          # update probabilities every N iterations
    reaction: float = 0.25,          # smoothing for weight update (0.1~0.3 good)
    score_best_w: float = 6.0,       # reward: makes new global best
    score_improve_w: float = 2.0,    # reward: improves current
    score_accept_w: float = 0.5,     # reward: accepted but not improve
    reconstruct_trials: int = 3,     # how many reconstructions per iteration
):
    """
    Multi-period DR:
      - start from greedy schedule+assign (baseline)
      - iterate: destroy u -> reconstruct -> evaluate
      - keep BOTH:
          * current state (walk)      : (U_curr, score_curr)
          * best-so-far (incumbent)   : (U_best, m_best, best_score)
      - accept if not much worse than current (epsilon)
      - adaptive destroy selection (ALNS-style) optional
      - logs per-iteration data into DR_log
      - logs per-batch data into DR_batches (every update_every iters)
    """
    import time
    import numpy as np
    import pandas as pd

    # -------------------------
    # RNG
    # -------------------------
    rng = np.random.default_rng(seed)

    coords_I, coords_J = inst["coords_I"], inst["coords_J"]
    demand_IT = inst["demand_IT"]
    M = len(coords_I)
    N = len(coords_J)

    # arcs
    distIJ, in_range, Ji, Ij = build_arcs(coords_I, coords_J, D=D, forbid_self=False)

    # template model (no solving)
    m_template = build_multi_period_model(
        M=M, N=N, T=T,
        in_range=in_range, Ji=Ji, Ij=Ij,
        demand_IT=demand_IT, Q=Q, P_T=P_T,
        distIJ=distIJ, method_name=policy,
        max_chargers_per_site=max_chargers_per_site,
        cumulative_install=cumulative_install,
    )

    farther_of = compute_farther(distIJ, in_range, Ji)
    m_template = apply_method_multi(
        m_template, policy, distIJ, in_range, Ji, Ij, farther_of, verbose=False
    )

    # -------------------------
    # initial = greedy baseline
    # -------------------------
    base_out = run_one_policy_multi(
        inst=inst, policy=policy, P_T=P_T, Q=Q, D=D, T=T,
        exact_time_limit=exact_time_limit, exact_mip_gap=exact_mip_gap,
        max_chargers_per_site=max_chargers_per_site,
        cumulative_install=cumulative_install,
        seed=seed, verbose=verbose
    )

    m0 = base_out["m_best"]
    score0 = float(evaluate_solution_multi(m0, demand_IT)["covered_demand"])

    # current (walk)
    U_curr = _clone_u_matrix(m0)
    score_curr = float(score0)

    # best-so-far
    U_best = dict(U_curr)
    m_best = m0
    best_score = float(score0)

    # -------------------------
    # logs
    # -------------------------
    logger = DRLogger()
    dr_trace = []   # per-iteration rich diagnostics

    # per-batch logs
    batch_logs = []
    batch_start_best = float(best_score)

    # uniqueness tracking (optional)
    seen = set()
    def hash_u(Ud):
        return tuple(Ud[k] for k in sorted(Ud.keys()))
    seen.add(hash_u(U_curr))

    t_start = time.perf_counter()

    # -------------------------
    # Adaptive destroy init
    # -------------------------
    if adaptive_destroy:
        modes = list(destroy_modes)
        K_modes = len(modes)

        # start uniform probabilities
        p = np.ones(K_modes, dtype=float) / K_modes

        # ALNS weights (persist across windows)
        w = np.ones(K_modes, dtype=float)

        # window counters (reset every update_every)
        window_picked    = {m: 0 for m in modes}
        window_accepted  = {m: 0 for m in modes}
        window_impr_curr = {m: 0 for m in modes}
        window_impr_best = {m: 0 for m in modes}
        window_best_delta = {m: 0.0 for m in modes}
        window_reward_sum = {m: 0.0 for m in modes}

    else:
        modes = None
        p = None
        w = None

    # -------------------------
    # MAIN LOOP
    # -------------------------
    it = 0
    while it < int(max_iter) and (time.perf_counter() - t_start) < float(dr_time_limit):
        it += 1
        seed_iter = None if seed is None else (int(seed) + it)

        # 1) choose destroy mode
        if adaptive_destroy:
            mode = str(rng.choice(modes, p=p))
            window_picked[mode] += 1
        else:
            mode = str(destroy_mode)

        # 2) destroy CURRENT
        U_try, k_removed = destroy_multi_u(
            U_curr, P_T,
            frac_remove=frac_remove,
            mode=mode,
            seed=seed_iter,
            site_cap=max_chargers_per_site,
            cumulative_install=cumulative_install,
        )

        seen.add(hash_u(U_try))
        unique_count = len(seen)

        # 3) reconstruct multiple times, keep best local
        best_local_score = -1e18
        best_local_m = None
        best_local_U = None

        for _ in range(int(reconstruct_trials)):
            m_tmp = reconstruct_multi_u_greedy(
                m_template=m_template,
                Udict_partial=dict(U_try),
                distIJ=distIJ,
                demand_IT=demand_IT,
                P_T=P_T,
                policy=policy,
                cumulative_install=cumulative_install,
            )
            s_tmp = float(evaluate_solution_multi(m_tmp, demand_IT)["covered_demand"])
            if s_tmp > best_local_score:
                best_local_score = s_tmp
                best_local_m = m_tmp
                best_local_U = _clone_u_matrix(m_tmp)

        m_try = best_local_m
        score_try = float(best_local_score)
        U_try = best_local_U

        improved_curr = (score_try > score_curr + 1e-9)
        improved_best = (score_try > best_score + 1e-9)

        # 4) accept/reject
        accepted = (score_try >= score_curr - float(accept_epsilon))

        # rewards computed only if accepted (common ALNS choice)
        reward = 0.0

        if accepted:
            score_curr = score_try
            U_curr = _clone_u_matrix(m_try)
            seen.add(hash_u(U_curr))

            if improved_best:
                delta_best = score_try - best_score
                best_score = score_try
                m_best = m_try
                U_best = dict(U_curr)
            else:
                delta_best = 0.0

            if adaptive_destroy:
                window_accepted[mode] += 1
                if improved_curr:
                    window_impr_curr[mode] += 1
                if improved_best:
                    window_impr_best[mode] += 1
                    window_best_delta[mode] += float(delta_best)

                if improved_best:
                    reward = float(score_best_w)
                elif improved_curr:
                    reward = float(score_improve_w)
                else:
                    reward = float(score_accept_w)

                window_reward_sum[mode] += reward

        # 5) update probabilities every update_every
        if adaptive_destroy and (it % int(update_every) == 0):
            # batch row (so you can print table)
            batch_row = {
                "iter_from": it - int(update_every) + 1,
                "iter_to": it,
                "best_start": float(batch_start_best),
                "best_end": float(best_score),
                "batch_best_improvement": float(best_score - batch_start_best),
            }

            # update weights from window performance
            # score for mode = avg reward per pick (0 if never picked)
            for idx_m, m in enumerate(modes):
                picked = int(window_picked[m])
                acc = int(window_accepted[m])
                ic = int(window_impr_curr[m])
                ib = int(window_impr_best[m])
                bd = float(window_best_delta[m])
                rs = float(window_reward_sum[m])

                batch_row[f"{m}_picked"] = picked
                batch_row[f"{m}_accepted"] = acc
                batch_row[f"{m}_acc_rate"] = (acc / picked) if picked > 0 else 0.0
                batch_row[f"{m}_impr_curr"] = ic
                batch_row[f"{m}_impr_best"] = ib
                batch_row[f"{m}_best_delta"] = bd
                batch_row[f"{m}_reward_sum"] = rs
                batch_row[f"{m}_avg_reward"] = (rs / picked) if picked > 0 else 0.0
                batch_row[f"p_{m}"] = float(p[idx_m])

                score_mode = (rs / picked) if picked > 0 else 0.0

                # ALNS smoothing update
                w[idx_m] = (1.0 - float(reaction)) * w[idx_m] + float(reaction) * score_mode
                w[idx_m] = max(w[idx_m], 1e-9)

            # new probabilities from updated weights
            p = w / w.sum()

            # store batch, reset window counters
            batch_logs.append(batch_row)
            batch_start_best = float(best_score)

            for m in modes:
                window_picked[m] = 0
                window_accepted[m] = 0
                window_impr_curr[m] = 0
                window_impr_best[m] = 0
                window_best_delta[m] = 0.0
                window_reward_sum[m] = 0.0

        # 6) core log (DRLogger strict fields only)
        elapsed_now = time.perf_counter() - t_start
        logger.log(
            it=it,
            score=score_try,
            best=best_score,
            elapsed=elapsed_now,
            k_remove=k_removed,
            mode=mode,
            accepted=accepted,
            unique=unique_count,
        )

        # 7) rich trace (your notebook can plot probability dynamics)
        row = {
            "iteration": it,
            "score": score_try,
            "best": best_score,
            "score_curr": score_curr,
            "elapsed": elapsed_now,
            "k_removed": k_removed,
            "mode": mode,
            "accepted": accepted,
            "unique": unique_count,
        }
        if adaptive_destroy:
            for idx_m, m in enumerate(modes):
                row[f"p_{m}"] = float(p[idx_m])
                row[f"w_{m}"] = float(w[idx_m])
        dr_trace.append(row)

    return dict(
        policy=policy,
        score_start=float(score0),
        score_best=float(best_score),
        DR_log=logger.to_df(),                # clean curve
        DR_trace=pd.DataFrame(dr_trace),      # per-iteration rich info
        DR_batches=pd.DataFrame(batch_logs),  # per-window table
        m_best=m_best,
        distIJ=distIJ,
        DR_batches=pd.DataFrame(batch_logs),
    )
