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

def pick_site_topk(scores_dict, k=10, rng=None):
    """
    scores_dict: {j: score}, higher is better
    returns: one j sampled uniformly from top-k
    """
    if rng is None:
        rng = np.random.default_rng()
    items = sorted(scores_dict.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return None
    top = items[: min(k, len(items))]
    js = [j for j, _ in top]
    return int(rng.choice(js))

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
    # NEW knobs to weaken greedy (good for DR curve)
    greedy_topk: int = 1.0,
    greedy_noise: float = 0.0,
):
    """
    Multi-period baseline runner (Priority-1/2):
      - Exact (time-limited)
      - Greedy schedule (u) + greedy fractional assignment (y)

    Notes:
      - Greedy is intentionally weakened via noisy top-K so DR has room to improve.
      - Keeps fractional assignment (y in [0,1]).
    """
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

    rng = np.random.default_rng(seed)

    coords_I, coords_J = inst["coords_I"], inst["coords_J"]
    demand_IT = inst["demand_IT"]  # list length T, each length M
    M = len(coords_I)
    N = len(coords_J)

    # arcs
    distIJ, in_range, Ji, Ij = build_arcs(
        coords_I, coords_J, D=D, forbid_self=forbid_self
    )

    def _extract_mip_gap_and_bound(res, sense: str = "max"):
        """
        Returns (gap, best_bound, incumbent_obj) if available, else (None, None, None).

        sense: "max" or "min" (your model is MAX covered_demand)
        """
        gap = None
        best_bound = None
        incumbent = None

        # 1) Try direct fields (rarely works, but cheap)
        try:
            gap = getattr(res.solver, "mip_gap", None)
            if gap is None:
                gap = getattr(res.solver, "gap", None)
        except Exception:
            pass

        # 2) Try solver backend model (common for Gurobi / persistent interfaces)
        sm = None
        try:
            sm = getattr(res.solver, "_solver_model", None)
        except Exception:
            sm = None

        # ---- GUROBI ----
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

            # ---- CPLEX (sometimes) ----
            try:
                # some CPLEX objects expose relative gap via solution.MIP
                if gap is None and hasattr(sm, "solution") and hasattr(sm.solution, "MIP"):
                    gap = float(sm.solution.MIP.get_mip_relative_gap())
                if best_bound is None and hasattr(sm, "solution") and hasattr(sm.solution, "MIP"):
                    best_bound = float(sm.solution.MIP.get_best_objective())
            except Exception:
                pass

        # 3) If we have bound + incumbent but no gap, compute it
        if gap is None and (best_bound is not None) and (incumbent is not None):
            denom = max(1.0, abs(incumbent))
            if sense.lower().startswith("max"):
                # for MAX: bound >= incumbent
                gap = max(0.0, (best_bound - incumbent) / denom)
            else:
                # for MIN: incumbent >= bound
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

        # termination & proven optimal
        tc = None
        try:
            tc = res.solver.termination_condition
        except Exception:
            tc = getattr(res, "termination_condition", None)

        exact_termination = tc
        proven_optimal_exact = (tc == TerminationCondition.optimal)

        # gap (solver-dependent; may be None)
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

    # clear state (important)
    for t in m_g.T:
        for j in m_g.J:
            m_g.u[j, t].value = 0
            m_g.x[j, t].value = 0
            m_g.z[j, t].value = 0

    # U cap (prefer model’s U)
    U = int(m_g.U.value) if hasattr(m_g, "U") else (int(max_chargers_per_site) if max_chargers_per_site is not None else int(sum(P_T)))

    # build Ij list for reach scoring
    Ij_int = {}
    for (i, j) in in_range:
        Ij_int.setdefault(int(j), []).append(int(i))

    def x_now(j, t_int):
        """charger count at (j,t) implied by current u (respect cumulative flag)."""
        if cumulative_install:
            return sum(int(m_g.u[j, tt].value or 0) for tt in m_g.T if int(tt) <= int(t_int))
        return int(m_g.u[j, t_int].value or 0)

    t0 = time.perf_counter()

    # schedule installs (NOISY top-K greedy)
    TOPK = max(1, int(greedy_topk))
    NOISE = float(greedy_noise)

    for t in range(T):
        for _ in range(int(P_T[t])):

            scored = []
            for j in m_g.J:
                if x_now(j, t) >= U:
                    continue

                reach = sum(float(demand_IT[t][i]) for i in Ij_int.get(int(j), []))
                base = reach / (1.0 + x_now(j, t))

                # noise to weaken greedy (gives DR room)
                s = base * (1.0 + NOISE * rng.normal())
                scored.append((s, int(j)))

            if not scored:
                break

            scored.sort(reverse=True, key=lambda x: x[0])
            k = min(TOPK, len(scored))
            cand_js = [j for (_, j) in scored[:k]]
            best_j = int(rng.choice(cand_js))

            m_g.u[best_j, t].value = int(m_g.u[best_j, t].value or 0) + 1

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
):
    """
    Strong destroy operators on Udict only.

    Returns:
        (U_new, k_removed)

    modes:
      - "k_units": remove k units globally (k ~ frac_remove * total_u)
      - "site_all": pick 1 site j*, set u[j*,t]=0 for all t
      - "site_future": pick site j* and time t0, set u[j*,t]=0 for t>=t0
      - "random_unit": old behavior (weak), remove single units proportional to u
    """
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
    Js = sorted({j for (j, t) in U_new.keys()})
    Ts = sorted({t for (j, t) in U_new.keys()})

    # ---------- site_all ----------
    if mode in ("site_all", "site"):
        # choose only sites that currently have at least 1 unit somewhere
        candidates = []
        for j in Js:
            totj = sum(int(U_new[(int(j), int(t))]) for t in Ts)
            if totj > 0:
                candidates.append(int(j))

        if not candidates:
            return U_new, 0

        j_star = int(np.random.choice(candidates))
        before = sum(int(U_new[(j_star, int(t))]) for t in Ts)
        for t in Ts:
            U_new[(j_star, int(t))] = 0
        return U_new, before


    # ---------- site_future ----------
    if mode in ("site_future", "future"):
        # choose only sites that have some units
        candidates = []
        for j in Js:
            totj = sum(int(U_new[(int(j), int(t))]) for t in Ts)
            if totj > 0:
                candidates.append(int(j))

        if not candidates:
            return U_new, 0

        j_star = int(np.random.choice(candidates))
        t0 = int(np.random.choice(Ts))

        before = sum(int(U_new[(j_star, int(t))]) for t in Ts if int(t) >= t0)
        if before <= 0:
            return U_new, 0

        for t in Ts:
            if int(t) >= t0:
                U_new[(j_star, int(t))] = 0
        return U_new, before


    # decide k for unit-based destroys
    if mode in ("k_units", "k"):
        k_remove = max(1, int(np.ceil(frac_remove * total)))

    else:
        # "random_unit" (old weak behavior)
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
    reconstruct_mode: str = "rand_greedy",   # NEW
    topK: int = 8,                           # NEW
    noise: float = 0.03,                     # NEW
):
        # reproducible randomness for reconstruction
    rng = np.random.default_rng()

    """
    Reconstruction step:
      - Start from partial u
      - Fill missing installs period-by-period using the SAME greedy logic as your baseline
      - Then sync x,z and greedy assign y

    Returns a *model* (clone of template) with full u/x/z/y.
    """
    # Clone the model template (safe) so we don't mutate external object
    m = m_template.clone()

    # 1) set partial u
    _apply_u_matrix(m, Udict_partial)

    # 2) fill missing per period
    # compute current per-period sums
    T = len(P_T)

    # build Ij_int from arcs for reach scoring (same as your run_one_policy_multi)
    Ij_int = {}
    for (i, j) in m.Arcs:
        Ij_int.setdefault(int(j), []).append(int(i))

    U_cap = int(m.U.value) if hasattr(m, "U") else int(max(P_T))

    def x_now(j, t):
        # compute available chargers at (j,t) implied by current u in model
        if cumulative_install:
            return sum(int(m.u[j, tt].value or 0) for tt in m.T if int(tt) <= t)
        return int(m.u[j, t].value or 0)

    for t in range(T):
        # how many already installed in this period
        already = sum(int(m.u[j, t].value or 0) for j in m.J)
        missing = max(0, int(P_T[t]) - int(already))
        if missing <= 0:
            continue

        for _ in range(missing):
            # build scored candidate list
            cands = []
            for j in m.J:
                if x_now(j, t) >= U_cap:
                    continue
                reach = sum(float(demand_IT[t][i]) for i in Ij_int.get(int(j), []))
                base = reach / (1.0 + x_now(j, t))
                score = base * (1.0 + 0.15 * rng.normal())   # 15% noise

                if noise > 0:
                    score = score * (1.0 + noise * np.random.randn())
                cands.append((score, j))

            if not cands:
                break

            cands.sort(reverse=True, key=lambda x: x[0])

            rm = (reconstruct_mode or "rand_greedy").lower()
            if rm in ("greedy", "deterministic"):
                best_j = cands[0][1]
            else:
                # randomized greedy: pick among topK
                K = max(1, min(int(topK), len(cands)))
                idx = int(np.random.randint(0, K))
                best_j = cands[idx][1]


            if best_j is None:
                break

            m.u[best_j, t].value = int(m.u[best_j, t].value or 0) + 1

    # 3) sync x,z then assign y
    sync_solution_state(m, cumulative_install=cumulative_install)
    m = reassign_y_greedy_multi(m, distIJ, Ji=None, method_name=policy, cumulative_install=cumulative_install)
    # note: your reassign_y_greedy_multi rebuilds Ji internally from arcs :contentReference[oaicite:4]{index=4}

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
    destroy_mode: str = "random",
    exact_time_limit: float = 120,
    exact_mip_gap: float = 0.10,
    max_chargers_per_site: int | None = None,
    cumulative_install: bool = True,
    seed: int | None = None,
    verbose: bool = False,
    accept_epsilon: float = 0.02,
):
    """
    Multi-period DR:
      - start from greedy schedule+assign (baseline)
      - iterate: destroy u -> reconstruct (greedy) -> evaluate
      - keep BOTH:
          * current state (walk state)   : (U_curr, score_curr)
          * best-so-far (incumbent best) : (U_best, m_best, best_score)
      - accept if not much worse than current (epsilon)
      - update best only on strict improvement
      - log DR curve
    """
    if seed is not None:
        np.random.seed(seed)

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

    # --- initial = greedy baseline ---
    base_out = run_one_policy_multi(
        inst=inst, policy=policy, P_T=P_T, Q=Q, D=D, T=T,
        exact_time_limit=exact_time_limit, exact_mip_gap=exact_mip_gap,
        max_chargers_per_site=max_chargers_per_site,
        cumulative_install=cumulative_install,
        seed=seed, verbose=verbose
    )

    m0 = base_out["m_best"]
    score0 = evaluate_solution_multi(m0, demand_IT)["covered_demand"]

    # current (walk state)
    U_curr = _clone_u_matrix(m0)
    score_curr = score0

    # best-so-far (incumbent)
    U_best = dict(U_curr)
    m_best = m0
    best_score = score0

    # logging
    logger = DRLogger()
    t_start = time.perf_counter()

    seen = set()
    def hash_u(Ud):
        return tuple(Ud[k] for k in sorted(Ud.keys()))

    seen.add(hash_u(U_curr))

    it = 0
    while it < max_iter and (time.perf_counter() - t_start) < dr_time_limit:
        it += 1

        # 1) destroy CURRENT (not best), returns (U_try, k_removed)
        # iteration-dependent seed (or None)
        seed_iter = None if seed is None else (seed + it)

        U_try, k_removed = destroy_multi_u(
            U_curr,
            P_T,
            frac_remove=frac_remove,
            mode=destroy_mode,
            seed=seed_iter,     # <<< FIX
        )


        # uniqueness count for the tried solution
        seen.add(hash_u(U_try))
        unique_count = len(seen)

        # 2) reconstruct K times → keep the best
        # 2) reconstruct K times -> keep best (no reseed here)
        K = 8
        best_local_score = -1e18
        best_local_m = None
        best_local_U = None

        for r in range(K):
            m_tmp = reconstruct_multi_u_greedy(
                m_template=m_template,
                Udict_partial=dict(U_try),   # fresh copy every time
                distIJ=distIJ,
                demand_IT=demand_IT,
                P_T=P_T,
                policy=policy,
                cumulative_install=cumulative_install,
            )
            s_tmp = evaluate_solution_multi(m_tmp, demand_IT)["covered_demand"]

            if s_tmp > best_local_score:
                best_local_score = s_tmp
                best_local_m = m_tmp
                best_local_U = _clone_u_matrix(m_tmp)

        m_try = best_local_m
        score_try = best_local_score
        U_try = best_local_U


        # 4) acceptance rule (epsilon-worse allowed relative to CURRENT)
        accepted = (score_try >= score_curr - float(accept_epsilon))

        if accepted:
            # move the walk
            U_curr = _clone_u_matrix(m_try)
            score_curr = score_try
            seen.add(hash_u(U_curr))  # current is also visited

            # update incumbent best only on strict improvement
            if score_try > best_score + 1e-9:
                best_score = score_try
                m_best = m_try
                U_best = dict(U_curr)

        # 5) log
        logger.log(
            it=it,
            score=score_try,
            best=best_score,
            elapsed=time.perf_counter() - t_start,
            k_remove=k_removed,
            mode=destroy_mode,
            accepted=accepted,
            unique=unique_count,
        )

    return dict(
        policy=policy,
        score_start=base_out["score_greedy"],
        score_best=best_score,
        DR_log=logger.to_df(),
        m_best=m_best,
        distIJ=distIJ,
    )
