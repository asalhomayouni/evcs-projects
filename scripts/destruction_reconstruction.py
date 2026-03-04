import numpy as np
import pandas as pd
import time
from pathlib import Path
from pyomo.opt import TerminationCondition

from evcs.geom import build_arcs
from evcs.model import build_base_model ,build_multi_period_model
from evcs.methods import (
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
    greedy_init_simple_variants,
    greedy_schedule_multi_from_variants,
    evaluate_policy_objective_multi, 

)
from evcs.solve import solve_model
from pyomo.environ import value
from Instance import generate_instance, save_instance_npz as save_instance, load_instance_npz as load_instance


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
            mip_gap=exact_mip_gap
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
        x_init = greedy_init_simple_variants(
            inst,
            B=P,              # total chargers budget (if P is total chargers)
            D_cover=D,        # your coverage distance
            variant=greedy_init_mode,  # map your mode string to variants
            gamma=0.5,
            tau=0.6*D,
            cooldown_steps=3,
            beta=0.10,
            topK=5,
            seed=seed,
        )

        m_greedy = m0

        # Write x and z
        for j in range(N):
            m_greedy.x[j].value = int(x_init[j])
            m_greedy.z[j].value = 1 if x_init[j] > 0 else 0

        # Optional: build y assignment (if your evaluation expects y)
        # If you already have a greedy assign function:
        # reassign_y_greedy(m_greedy, distIJ, demand_I, Q, D, ...)

    except Exception as e:
        print(f"⚠️ Simplified greedy init failed, using base model: {e}")
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
    greedy_variant="ring",
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
    # assumes you already have these helpers in your project
    # build_arcs, solve_model, and your greedy constructor (whatever you already use)

    coords_I, coords_J = inst["coords_I"], inst["coords_J"]
    M = len(coords_I)
    N = len(coords_J)

    # -------------------------
    # Demand canonicalization
    # model wants demand_IT[t][i] => shape (T, M)
    # keep also demand_MT for any numpy eval if needed elsewhere
    # -------------------------
    demand_arr = np.asarray(inst["demand_IT"], dtype=float)
    if demand_arr.ndim != 2:
        raise ValueError(f"demand_IT must be 2D, got shape={demand_arr.shape}")

    if demand_arr.shape == (T, M):
        demand_TM = demand_arr
        demand_MT = demand_arr.T
    elif demand_arr.shape == (M, T):
        demand_TM = demand_arr.T
        demand_MT = demand_arr
    else:
        # try to infer T if user passed stale T
        if demand_arr.shape[1] == M:
            demand_TM = demand_arr
            demand_MT = demand_arr.T
            T = int(demand_TM.shape[0])
        elif demand_arr.shape[0] == M:
            demand_MT = demand_arr
            demand_TM = demand_arr.T
            T = int(demand_TM.shape[0])
        else:
            raise ValueError(f"demand_IT shape {demand_arr.shape} incompatible with M={M}")

    # align P_T length to T
    P_T = list(P_T)
    if len(P_T) > T:
        P_T = P_T[:T]
    elif len(P_T) < T:
        P_T = P_T + [P_T[-1]] * (T - len(P_T))

    # -------------------------
    # arcs
    # -------------------------
    distIJ, in_range, Ji, Ij = build_arcs(coords_I, coords_J, D=D, forbid_self=forbid_self)

    # -------------------------
    # helpers
    # -------------------------
    def _extract_mip_gap_and_bound(res, sense: str = "max"):
        gap = None
        best_bound = None
        incumbent = None

        # best-effort from solver internals (may be None depending on backend)
        try:
            sm = getattr(res.solver, "_solver_model", None)
        except Exception:
            sm = None

        if sm is not None:
            for attr, target in [("MIPGap", "gap"), ("ObjBound", "bound"), ("ObjVal", "inc")]:
                try:
                    val = getattr(sm, attr, None)
                    if val is not None:
                        if target == "gap":
                            gap = float(val)
                        elif target == "bound":
                            best_bound = float(val)
                        else:
                            incumbent = float(val)
                except Exception:
                    pass

        # fallback compute if possible
        if gap is None and (best_bound is not None) and (incumbent is not None):
            denom = max(1.0, abs(incumbent))
            if sense.lower().startswith("max"):
                gap = max(0.0, (best_bound - incumbent) / denom)
            else:
                gap = max(0.0, (incumbent - best_bound) / denom)

        return gap, best_bound, incumbent

    def _any_incumbent_loaded(m):
        # robust: check if any u variable has a numeric value
        try:
            for k in m.u:
                if m.u[k].value is not None:
                    return True
        except Exception:
            pass
        return False

    # -------------------------
    # outputs (greedy placeholders)
    # -------------------------
    m_greedy = None
    score_greedy = None

    # EXACT solve
    # -------------------------
    m_exact = None
    score_exact = None
    time_exact = None
    exact_termination = None
    proven_optimal_exact = False
    exact_gap = None
    exact_bound = None
    exact_incumbent_obj = None
    exact_has_feasible = False

    try:
        m_exact = build_multi_period_model(
            M=M, N=N, T=T,
            in_range=in_range, Ji=Ji, Ij=Ij,
            demand_IT=demand_TM,      # (T,M) ✅
            Q=Q, P_T=P_T,
            distIJ=distIJ,
            method_name=policy,
            max_chargers_per_site=max_chargers_per_site,
            cumulative_install=cumulative_install,
        )

        farther_of = compute_farther(distIJ, in_range, Ji)
        m_exact = apply_method_multi(m_exact, policy, distIJ, in_range, Ji, Ij, farther_of, verbose=False)
        
        import time
        t0 = time.perf_counter()
        
        # ✅ Let solve_model load the incumbent if it can.
        # Your solve.py supports load_solution, so use it.
        sol = solve_model(
            m_exact,
            verbose=verbose,
            time_limit=exact_time_limit,
            mip_gap=exact_mip_gap,
            load_solution=False,   # recommended for time-limited MIP
        )

        res = sol["res"]
        import numpy as np

        best_feas  = getattr(res, "best_feasible_objective", None)
        best_bound = getattr(res, "best_objective_bound", None)

        def _num(x):
            try:
                return float(x)
            except Exception:
                return np.nan

        best_feas_f  = _num(best_feas)
        best_bound_f = _num(best_bound)

        exact_has_feasible = np.isfinite(best_feas_f)

        # relative gap (if both exist)
        exact_gap = np.nan
        if exact_has_feasible and np.isfinite(best_bound_f):
            denom = max(1e-9, abs(best_feas_f))
            exact_gap = (best_bound_f - best_feas_f) / denom  # (for maximize, bound >= feas)

        # termination
        exact_term = getattr(res, "termination_condition", None)
        best_feas = sol.get("best_feasible_objective", None)
        best_bound = sol.get("best_objective_bound", None)

        exact_gap = None
        if best_feas is not None and best_bound is not None:
            denom = max(1e-9, abs(best_feas))
            exact_gap = (best_bound - best_feas) / denom   # maximize => bound >= feas

        time_exact = time.perf_counter() - t0
        
        # termination
        try:
            tc = res.solver.termination_condition
        except Exception:
            tc = getattr(res, "termination_condition", None)

        exact_termination = tc
        proven_optimal_exact = (tc == TerminationCondition.optimal)

        exact_gap, exact_bound, exact_incumbent_obj = _extract_mip_gap_and_bound(res, sense="max")

        # ✅ Robust feasibility/incumbent detection:
        # If solver loaded vars (even under time limit), treat as feasible.
        if _any_incumbent_loaded(m_exact):
            exact_has_feasible = True
            # score with loaded values
            score_exact = float(evaluate_solution_multi(m_exact, demand_TM)["covered_demand"])
        else:
            # no incumbent loaded => no exact score
            exact_has_feasible = False
            score_exact = None
            if verbose:
                print(f"[Exact] No incumbent values loaded. termination={tc}")

    except Exception as e:
        if verbose:
            print("[Exact] failed:", e)
        m_exact = None
        score_exact = None
        exact_has_feasible = False

  
    # -------------------------
    # GREEDY schedule + greedy assign
    # -------------------------
    m_g = build_multi_period_model(
        M=M, N=N, T=T,
        in_range=in_range, Ji=Ji, Ij=Ij,
        demand_IT=demand_TM, Q=Q, P_T=P_T,
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
    # FAST GREEDY SCHEDULING (variant-based) via helper
    # =========================

    # choose variant (you can pass this in as an argument instead of hardcoding)
    greedy_variant = "ring"   # ring | exp | tabu | additive | topk_random

    U_cap = int(m_g.U.value) if hasattr(m_g, "U") else (
        int(max_chargers_per_site) if max_chargers_per_site is not None else None
    )

    U0 = greedy_schedule_multi_from_variants(
        inst=inst,
        P_T=P_T,
        D_cover=D,
        variant=greedy_variant,
        gamma=0.5,
        tau=0.6 * D,
        cooldown_steps=3,
        beta=0.10,
        topK=5,
        seed=seed or 0,
        cumulative_install=cumulative_install,
        U=U_cap,
        mode="aggregate_then_fill",   # or "per_period"
    )

    # write schedule into model
    for (j, t), val in U0.items():
        m_g.u[j, t].value = int(val)

    # sync x/z from u then assign y (KEEP YOUR EXISTING CODE)
    sync_solution_state(m_g, cumulative_install=cumulative_install)
    m_g = reassign_y_greedy_multi(m_g, distIJ, Ji, method_name=policy, cumulative_install=cumulative_install)

    score_greedy = float(evaluate_solution_multi(m_g, demand_TM)["covered_demand"])
    time_greedy = time.perf_counter() - t0

    return dict(
        policy=policy,
        greedy_variant=greedy_variant,
        m_best=m_g,
        m_greedy=m_g,
        score_greedy=score_greedy,
        m_exact=m_exact,
        score_exact=score_exact,
        proven_optimal_exact=bool(proven_optimal_exact),
        time_exact=time_exact,
        distIJ=distIJ,
        in_range=in_range,
        Ji=Ji,
        Ij=Ij,
        exact_has_feasible=bool(exact_has_feasible),
        exact_incumbent_obj=best_feas_f if exact_has_feasible else None,
        exact_bound=best_bound_f if np.isfinite(best_bound_f) else None,
        exact_gap=exact_gap if np.isfinite(exact_gap) else None,
        exact_termination=exact_term,


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

import numpy as np

def destroy_multi_u(
    Udict,
    inst,
    rng,
    P_T,
    frac_remove: float = 0.20,
    mode: str = "site_swap",
    seed: int | None = None,
    site_cap: int | None = None,
    cumulative_install: bool = True,
    area_radius: float | None = None,
    area_quantile: float = 0.25,
    local_mix=(0.50, 0.25, 0.25),  # k_units, site_all, site_future
):
   
    # Local RNG if seed provided (do NOT touch global np.random)
    if seed is not None:
        rng = np.random.default_rng(int(seed))

    mode = (mode or "site_swap").lower().strip()

    # Work on a COPY (do NOT mutate input)
    U_new = {k: int(v) for k, v in Udict.items()}

    keys = list(U_new.keys())
    if not keys:
        return U_new, 0

    total = sum(int(v) for v in U_new.values())
    if total <= 0:
        return U_new, 0

    # infer sets
    Js = sorted({int(j) for (j, t) in U_new.keys()})
    Ts = sorted({int(t) for (j, t) in U_new.keys()})
    T_max = max(Ts) if Ts else 0

    # helpers
    def tot_by_j():
        return {j: sum(int(U_new[(j, t)]) for t in Ts) for j in Js}

    def cum_in_site(j, t):
        """cumulative installs at site j up to period t (inclusive), based on U_new."""
        return sum(int(U_new[(j, tt)]) for tt in Ts if int(tt) <= int(t))

    # ---------------------------------------------------------
    # 1) SITE SWAP  (move schedule from one open site to another)
    # ---------------------------------------------------------
    if mode in ("site_swap"):
        totj = tot_by_j()
        open_sites = [j for j, v in totj.items() if v > 0]
        if not open_sites:
            return U_new, 0

        closed_sites = [j for j, v in totj.items() if v == 0]

        j_out = int(rng.choice(open_sites))

        # choose j_in != j_out, prefer closed
        if closed_sites:
            cand = [j for j in closed_sites if j != j_out]
            if not cand:
                cand = [j for j in Js if j != j_out]
        else:
            cand = [j for j in Js if j != j_out]

        if not cand:
            return U_new, 0

        j_in = int(rng.choice(cand))

        # move schedule period-by-period, enforcing cap if requested
        for t in Ts:
            v = int(U_new[(j_out, t)])
            if v <= 0:
                continue

            # compute how much we can add to j_in at this period
            add = v
            if site_cap is not None:
                if cumulative_install:
                    cur = cum_in_site(j_in, t)
                    remaining = max(0, int(site_cap) - cur)
                    add = min(add, remaining)
                else:
                    remaining = max(0, int(site_cap) - int(U_new[(j_in, t)]))
                    add = min(add, remaining)

            if add > 0:
                U_new[(j_in, t)] = int(U_new[(j_in, t)]) + add

            # remove from j_out regardless (destroy)
            U_new[(j_out, t)] = 0

        # site_swap is a move, not a removal
        return U_new, 0

    # ---------------------------------------------------------
    # 2) LOCAL REMOVE (merged: k_units + site_all + site_future)
    # ---------------------------------------------------------
    if mode == "local_remove":
        totj = tot_by_j()
        open_sites = [j for j, v in totj.items() if v > 0]
        if not open_sites:
            return U_new, 0

        j0 = int(rng.choice(open_sites))

        # choose which subtype to apply
        sub = rng.choice(["k_units", "site_all", "site_future"], p=list(local_mix))

        if sub == "site_all":
            removed = 0
            for t in Ts:
                v = int(U_new[(j0, t)])
                if v > 0:
                    removed += v
                    U_new[(j0, t)] = 0
            return U_new, removed

        if sub == "site_future":
            t_start = int(rng.choice(Ts))
            removed = 0
            for t in Ts:
                if int(t) >= t_start:
                    v = int(U_new[(j0, t)])
                    if v > 0:
                        removed += v
                        U_new[(j0, t)] = 0
            return U_new, removed

        # sub == "k_units"
        periods_with = [t for t in Ts if int(U_new[(j0, t)]) > 0]
        if not periods_with:
            return U_new, 0

        t0 = int(rng.choice(periods_with))
        v = int(U_new[(j0, t0)])
        k = max(1, int(round(float(frac_remove) * v)))
        k = min(k, v)

        U_new[(j0, t0)] = v - k
        return U_new, k

    # ---------------------------------------------------------
    # 3) AREA DESTROY (remove installs within radius across all periods)
    # ---------------------------------------------------------
    if mode == "area_destroy":
        coords_J = np.asarray(inst["coords_J"], dtype=float)

        totj = tot_by_j()
        active = [j for j, v in totj.items() if v > 0]
        if not active:
            return U_new, 0

        j_center = int(rng.choice(active))
        center_xy = coords_J[j_center]

        dists = np.sqrt(np.sum((coords_J - center_xy) ** 2, axis=1))

        # pick radius
        if area_radius is None:
            radius = float(np.quantile(dists, float(area_quantile)))
            if radius <= 1e-12:
                radius = float(np.max(dists)) * 0.10
        else:
            radius = float(area_radius)

        J_neigh = [int(j) for j in range(len(coords_J)) if float(dists[j]) <= radius]

        removed = 0
        for j in J_neigh:
            for t in Ts:
                v = int(U_new[(j, t)])
                if v > 0:
                    removed += v
                    U_new[(j, t)] = 0

        return U_new, removed

    # ---------------------------------------------------------
    raise ValueError(f"Unknown destroy mode: {mode}")



def reconstruct_multi_u_greedy(
    m_template,
    Udict_partial,
    distIJ,
    demand_IT,
    P_T,
    policy: str,
    cumulative_install: bool = True,
    top_k_choice: int = 3,      # ✅ NEW: pick among top-k
    p_elite: float = 0.85,      # ✅ NEW: choose best with prob p_elite
    rng=None,                   # ✅ NEW: pass rng from DR for reproducibility
):
    """
    Strong reconstruction (marginal uncovered demand), with optional elite-biased top-k randomness.

    Steps:
      1) clone template, apply partial u
      2) for each period t: add missing chargers using marginal uncovered demand scoring
         - capacity-aware uncovered update (serve up to Q per placed charger)
         - selection: pick best site with prob p_elite else random among top-k
      3) sync x/z then assign y via reassign_y_greedy_multi
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng(0)

    # Clone model so we don't mutate template
    m = m_template.clone()

    # 1) apply partial u
    _apply_u_matrix(m, Udict_partial)

    # Cache Q once
    Q = float(m.Q.value)

    # periods
    T = len(P_T)

    # Build Ij_int from arcs for reach scoring
    Ij_int = {}
    for (i, j) in m.Arcs:
        Ij_int.setdefault(int(j), []).append(int(i))

    # Per-site cap
    U_cap = int(m.U.value) if hasattr(m, "U") else int(max(P_T))

    # Make sure demand_IT is indexed by [t][i]
    # (Assuming you already canonicalized to (T, M))
    def demand_at(t, i):
        return float(demand_IT[t][i])

    def x_now(j, t):
        """Chargers at site j in period t (cumulative or per-period)."""
        j = int(j)
        t = int(t)
        if cumulative_install:
            s = 0
            for tt in m.T:
                if int(tt) <= t:
                    s += int(m.u[j, tt].value or 0)
            return s
        return int(m.u[j, t].value or 0)

    # Clamp params safely
    top_k_choice = int(top_k_choice) if top_k_choice is not None else 1
    top_k_choice = max(1, top_k_choice)
    p_elite = float(p_elite)
    p_elite = min(1.0, max(0.0, p_elite))

    for t in range(T):
        # how many already installed in period t?
        already = 0
        for j in m.J:
            already += int(m.u[int(j), int(t)].value or 0)

        missing = max(0, int(P_T[t]) - int(already))
        if missing <= 0:
            continue

        # uncovered nodes in this period (use indices of demand vector)
        M = len(demand_IT[t])
        uncovered = set(range(M))

        for _ in range(missing):
            cands = []
            # evaluate each candidate site j
            for j in m.J:
                jj = int(j)
                if x_now(jj, t) >= U_cap:
                    continue

                # marginal gain = sum of uncovered demand reachable from j
                s = 0.0
                for i in Ij_int.get(jj, []):
                    ii = int(i)
                    if ii in uncovered:
                        s += demand_at(t, ii)

                if s > 1e-12:
                    cands.append((float(s), jj))

            if not cands:
                break

            # sort by score desc
            cands.sort(key=lambda x: x[0], reverse=True)

            # elite-biased top-k selection
            k = min(top_k_choice, len(cands))
            topk = cands[:k]

            if rng.random() < p_elite:
                best_score, best_j = topk[0]
            else:
                best_score, best_j = topk[int(rng.integers(0, k))]

            if best_score <= 1e-12:
                break

            # install one charger at (best_j, t)
            m.u[best_j, int(t)].value = int(m.u[best_j, int(t)].value or 0) + 1

            # capacity-aware uncovered update: mark nodes served up to Q
            cap = Q
            reach_nodes = [int(i) for i in Ij_int.get(int(best_j), []) if int(i) in uncovered]
            reach_nodes.sort(key=lambda i: demand_at(t, i), reverse=True)

            used = 0.0
            for i in reach_nodes:
                di = demand_at(t, i)
                if used + di <= cap + 1e-9:
                    uncovered.discard(int(i))
                    used += di

    # 3) sync x,z then assign y
    sync_solution_state(m, cumulative_install=cumulative_install)
    m = reassign_y_greedy_multi(
        m, distIJ, Ji=None, method_name=policy, cumulative_install=cumulative_install
    )

    return m


def reconstruct_u_dict_fast(
    U_partial: dict,           # keys (j,t) -> installs
    demand_IT,                 # list length T of demand arrays (len M)
    P_T,
    Ij_int: dict,              # j -> list of i in range
    U_cap: int,
    Q: float,
    rng=None,
    cumulative_install: bool = True,
    top_k_choice: int = 3,

):
    """
    Fast reconstruction on U-dict only (NO Pyomo).

    - Starts from a partial U (some installs already set).
    - Fills missing installs per period using marginal uncovered-demand scoring.
    - Uses capacity-aware uncovered update with charger capacity Q.
    - Returns:
        U_filled : dict mapping (j,t) -> installs
        proxy_score : float

    IMPORTANT:
      This function *does not* know the DR proxy metric (which depends on
      assignment/order over demand nodes). Therefore, the returned proxy_score
      is only an internal reconstruction heuristic signal.

      In run_DR_multi() we compute the real proxy with
      evaluate_u_numpy_greedy_binary() and select the best trial using that
      consistent proxy.
    """
    import numpy as np

    if rng is None:
        rng = np.random.default_rng(0)

    # Copy (do not mutate caller)
    U = dict(U_partial)

    # Robustly infer J set:
    # Prefer Ij_int keys (all candidate sites); fallback to keys in U
    if Ij_int and len(Ij_int) > 0:
        Js = sorted({int(j) for j in Ij_int.keys()})
    else:
        Js = sorted({int(j) for (j, t) in U.keys()})

    T = len(P_T)

    def u_get(j, t):
        return int(U.get((int(j), int(t)), 0))

    def u_set(j, t, v):
        U[(int(j), int(t))] = int(v)

    def x_now(j, t):
        """Chargers at site j in period t (cumulative or not)."""
        j = int(j)
        t = int(t)
        if cumulative_install:
            # sum u[j,0..t]
            s = 0
            for tt in range(t + 1):
                s += u_get(j, tt)
            return s
        return u_get(j, t)

    proxy_total = 0.0
    Q = float(Q)
    U_cap = int(U_cap)

    for t in range(T):
        # Period t: how many already installed?
        already = 0
        for j in Js:
            already += u_get(j, t)

        missing = max(0, int(P_T[t]) - int(already))
        if missing <= 0:
            continue

        # uncovered demand mask (bool list is faster than set in tight loops)
        M = len(demand_IT[t])
        uncovered = [True] * M

        for _ in range(missing):
            topk = []  # list of (score, j)

            # choose site by marginal uncovered demand (random among top-k)
            for j in Js:
                if x_now(j, t) >= U_cap:
                    continue

                s = 0.0
                neigh = Ij_int.get(int(j), [])
                for i in neigh:
                    ii = int(i)
                    if uncovered[ii]:
                        s += float(demand_IT[t][ii])

                if s <= 1e-12:
                    continue

                topk.append((float(s), int(j)))

            if not topk:
                break

            # keep only top-k by score (k small, this is fast enough)
            k = int(top_k_choice) if top_k_choice is not None else 1
            k = max(1, k)
            topk.sort(key=lambda x: x[0], reverse=True)
            topk = topk[:k]

            # random tie-break among top-k (weighted by score optional; here: uniform)
            p_elite = 0.85  # move this OUTSIDE the loop later if you want

            if rng.random() < p_elite:
                best_score, best_j = topk[0]
            else:
                best_score, best_j = topk[int(rng.integers(0, len(topk)))]



            # place one charger at (best_j, t)
            u_set(best_j, t, u_get(best_j, t) + 1)

            # capacity-aware uncovered update: greedily mark demand as served up to Q
            used = 0.0
            neigh = Ij_int.get(best_j, [])

            # collect reachable uncovered nodes then sort by demand descending
            reach = []
            for i in neigh:
                ii = int(i)
                if uncovered[ii]:
                    reach.append(ii)

            reach.sort(key=lambda ii: float(demand_IT[t][ii]), reverse=True)

            for ii in reach:
                di = float(demand_IT[t][ii])
                if used + di <= Q + 1e-9:
                    uncovered[ii] = False
                    used += di

            # NOTE: this is *not* the DR proxy metric; only a local signal.
            proxy_total += used

    return U, float(proxy_total)

def _u_to_capacity_array(U_dict, T, N, Q_cap, cumulative_install=True):
    """
    Convert U_dict[(t,j)] = chargers installed at period t at site j
    into cap[t,j] = demand-capacity (chargers * Q_cap), optionally cumulative over time.
    """
    import numpy as np
    cap = np.zeros((T, N), dtype=float)
    for (t, j), val in U_dict.items():
        t = int(t); j = int(j)
        if 0 <= t < T and 0 <= j < N:
            cap[t, j] += float(val) * float(Q_cap)
    if cumulative_install:
        cap = np.cumsum(cap, axis=0)
    return cap


def evaluate_u_numpy_greedy(
    U_dict,
    demand_IT,
    J_i_list,
    distIJ,
    Q_cap,
    T,
    N,
    cumulative_install=True,
    pre_sorted_J_i=None,   # optional: list of sorted feasible sites for each i
):
    """
    Greedy evaluation: allocate each user's demand to nearest feasible sites with remaining capacity.
    Returns covered_demand (float).

    - demand_IT assumed indexable like demand_IT[i, t]
    - distIJ can be dict[(i,j)] or array distIJ[i,j]
    - U_dict keys assumed (t,j) chargers installed at t at site j
    """


# -------------------------
# Helpers (put ABOVE run_DR_multi)
# -------------------------
import numpy as np

def _u_to_capacity_array(U_dict, T, N, Q_cap, cumulative_install=True):
    """
    U_dict[(j,t)] = chargers installed at period t at site j
    -> cap[t,j]   = capacity at period t at site j (chargers * Q_cap)
    """
    import numpy as np
    cap = np.zeros((T, N), dtype=float)

    for (j, t), val in U_dict.items():
        j = int(j); t = int(t)
        if 0 <= t < T and 0 <= j < N:
            cap[t, j] += float(val) * float(Q_cap)

    if cumulative_install:
        cap = np.cumsum(cap, axis=0)

    return cap

def evaluate_u_numpy_greedy(
    U_dict,
    demand_IT,
    J_i_list,
    distIJ,
    Q_cap,
    T,
    N,
    cumulative_install=True,
    pre_sorted_J_i=None,
):
    """
    Greedy evaluation: allocate each user's demand to nearest feasible sites with remaining capacity.
    Returns covered_demand (float).

    Notes:
    - demand_IT assumed indexable as demand_IT[i, t]
    - distIJ can be dict[(i,j)] or array distIJ[i,j]
    - U_dict keys assumed (t,j) chargers installed at t at site j
    """
    cap = _u_to_capacity_array(U_dict, T=T, N=N, Q_cap=Q_cap, cumulative_install=cumulative_install)

    covered = 0.0
    M = len(J_i_list)
    is_dict = isinstance(distIJ, dict)

    for t in range(T):
        cap_t = cap[t].copy()

        for i in range(M):
            d = float(demand_IT[i, t]) if isinstance(demand_IT, list) else float(demand_IT[i, t])

            if d <= 1e-12:
                continue

            js = J_i_list[i]
            if not js:
                continue

            if pre_sorted_J_i is not None:
                js_sorted = pre_sorted_J_i[i]
            else:
                js_sorted = sorted(
                    js,
                    key=lambda j: distIJ[(i, j)] if is_dict else distIJ[i, j]
                )

            remaining = d
            for j in js_sorted:
                if remaining <= 1e-12:
                    break
                avail = cap_t[j]
                if avail <= 1e-12:
                    continue
                take = avail if avail < remaining else remaining
                cap_t[j] -= take
                remaining -= take
                covered += take

    return float(covered)

import numpy as np

def _u_to_capacity_array_jt(U_dict, T, N, Q_cap, cumulative_install=True):
    """
    U_dict[(j,t)] = chargers installed at period t at site j
    -> cap[t,j] = (chargers * Q_cap), optionally cumulative over time.
    """
    cap = np.zeros((T, N), dtype=float)
    for (j, t), val in U_dict.items():
        j = int(j); t = int(t)
        if 0 <= t < T and 0 <= j < N:
            cap[t, j] += float(val) * float(Q_cap)

    if cumulative_install:
        cap = np.cumsum(cap, axis=0)
    return cap


def evaluate_u_numpy_greedy_jt(
    U_dict,
    demand_MT,        # (M,T)
    pre_sorted_J_i,   # list length M, each is feasible sites sorted by dist
    Q_cap,
    T,
    N,
    cumulative_install=True,
):
    """
    Greedy multi-source allocation with capacities.
    Uses U_dict keys (j,t).
    Returns covered demand (float).
    """
    cap = _u_to_capacity_array_jt(U_dict, T=T, N=N, Q_cap=Q_cap, cumulative_install=cumulative_install)

    covered = 0.0
    M = int(demand_MT.shape[0])

    for t in range(T):
        cap_t = cap[t].copy()

        for i in range(M):
            d = float(demand_MT[i, t])
            if d <= 1e-12:
                continue

            js_sorted = pre_sorted_J_i[i]
            if not js_sorted:
                continue

            remaining = d
            for j in js_sorted:
                if remaining <= 1e-12:
                    break
                avail = cap_t[j]
                if avail <= 1e-12:
                    continue
                take = avail if avail < remaining else remaining
                cap_t[j] -= take
                remaining -= take
                covered += take

    return float(covered)

def evaluate_u_numpy_greedy_binary(
    U_dict,
    demand_TM,          # shape (T, M)  demand_TM[t][i]
    J_i_list,           # list length M, each is list of feasible sites j
    distIJ,
    Q_cap,
    T,
    N,
    cumulative_install=True,
    pre_sorted_J_i=None
):
    """
    Greedy evaluation that matches the typical binary assignment idea:
    each (i,t) is either fully covered by ONE site or not covered.
    No demand splitting across multiple sites.

    Returns total covered demand (float).
    """
    import numpy as np

    cap = _u_to_capacity_array(U_dict, T=T, N=N, Q_cap=Q_cap, cumulative_install=cumulative_install)
    covered = 0.0
    M = len(J_i_list)
    is_dict = isinstance(distIJ, dict)

    for t in range(T):
        cap_t = cap[t].copy()

        for i in range(M):
            d = float(demand_TM[t][i])
            if d <= 1e-12:
                continue

            js = J_i_list[i]
            if not js:
                continue

            if pre_sorted_J_i is not None:
                js_sorted = pre_sorted_J_i[i]
            else:
                js_sorted = sorted(js, key=lambda j: distIJ[(i, j)] if is_dict else distIJ[i, j])

            # assign to first site with enough remaining capacity to cover full demand
            for j in js_sorted:
                if cap_t[j] + 1e-12 >= d:
                    cap_t[j] -= d
                    covered += d
                    break

    return float(covered)


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
    frac_remove_jump = 0.45,   # strong perturbation level (only used on plateau)
    destroy_mode: str = "local_remove",
    exact_time_limit: float = 120,
    exact_mip_gap: float = 0.10,
    max_chargers_per_site: int | None = None,
    cumulative_install: bool = True,
    seed: int | None = None,
    verbose: bool = False,
    accept_epsilon: float = 0.02,

    adaptive_destroy: bool = True,
    destroy_modes = ["site_swap", "local_remove", "area_destroy"],
    update_every: int = 25,
    reaction: float = 0.25,
    score_best_w: float = 6.0,
    score_improve_w: float = 2.0,
    score_accept_w: float = 0.5,
    reconstruct_trials: int = 3,

    # reconstruction diversity knobs
    top_k_choice: int = 3,
):
    import time
    import numpy as np
    import pandas as pd
    from pyomo.environ import value

    rng = np.random.default_rng(seed if seed is not None else 0)

    coords_I, coords_J = inst["coords_I"], inst["coords_J"]
    M = len(coords_I)
    N = len(coords_J)

    # -------------------------------------------------
    # Demand canonicalization ONCE -> demand_TM = (T,M)
    # -------------------------------------------------
    demand_raw = inst["demand_IT"]
    demand_arr = np.asarray(demand_raw, dtype=float)
    if demand_arr.ndim != 2:
        raise ValueError(f"demand_IT must be 2D. Got shape={getattr(demand_arr,'shape',None)}")

    if demand_arr.shape == (T, M):
        demand_TM = demand_arr
    elif demand_arr.shape == (M, T):
        demand_TM = demand_arr.T
    else:
        if demand_arr.shape[1] == M:
            demand_TM = demand_arr
            T = int(demand_TM.shape[0])
        elif demand_arr.shape[0] == M:
            demand_TM = demand_arr.T
            T = int(demand_TM.shape[0])
        else:
            raise ValueError(f"demand_IT shape {demand_arr.shape} incompatible with M={M} and T={T}")

    # Align P_T to final T
    P_T = list(P_T)
    if len(P_T) > T:
        P_T = P_T[:T]
    elif len(P_T) < T:
        P_T = P_T + [P_T[-1]] * (T - len(P_T))

    # -------------------------------------------------
    # Arcs + adjacency for reconstruction
    # -------------------------------------------------
    distIJ, in_range, Ji, Ij = build_arcs(coords_I, coords_J, D=D, forbid_self=False)

    Ij_int = {j: [] for j in range(N)}
    Ji_int = {i: [] for i in range(M)}
    for (i, j) in in_range:
        i = int(i); j = int(j)
        if 0 <= i < M and 0 <= j < N:
            Ij_int[j].append(i)
            Ji_int[i].append(j)

    # -------------------------------------------------
    # Pre-sort feasible sites for each demand node (closest-first)
    # -------------------------------------------------
    is_dict = isinstance(distIJ, dict)
    J_i_list = []
    for i in range(M):
        js = Ji_int.get(i, [])
        if not js:
            J_i_list.append([])
        else:
            if is_dict:
                J_i_list.append(sorted(js, key=lambda j: distIJ[(i, j)]))
            else:
                J_i_list.append(sorted(js, key=lambda j: distIJ[i, j]))
    pre_sorted_J_i = J_i_list

    # -------------------------------------------------
    # Template model (no solve)
    # -------------------------------------------------
    m_template = build_multi_period_model(
        M=M, N=N, T=T,
        in_range=in_range, Ji=Ji, Ij=Ij,
        demand_IT=demand_TM,                 # IMPORTANT: use (T,M)
        Q=Q, P_T=P_T,
        distIJ=distIJ, method_name=policy,
        max_chargers_per_site=max_chargers_per_site,
        cumulative_install=cumulative_install,
    )
    farther_of = compute_farther(distIJ, in_range, Ji)
    m_template = apply_method_multi(m_template, policy, distIJ, in_range, Ji, Ij, farther_of, verbose=False)

    U_cap = int(m_template.U.value) if hasattr(m_template, "U") else int(max(P_T))
    Q_cap = float(m_template.Q.value)

    # -------------------------------------------------
    # Initial greedy + exact passthrough
    # -------------------------------------------------
    base_out = run_one_policy_multi(
        inst=inst, policy=policy, P_T=P_T, Q=Q, D=D, T=T,
        exact_time_limit=exact_time_limit, exact_mip_gap=exact_mip_gap,
        max_chargers_per_site=max_chargers_per_site,
        cumulative_install=cumulative_install,
        seed=seed, verbose=verbose
    )

    # start from greedy solution U
    m0 = base_out["m_best"]
    U_curr = _clone_u_matrix(m0)

    # DR proxy state (fast, consistent with capacity/binary assignment)
    proxy_curr = float(evaluate_u_numpy_greedy_binary(
        U_curr,
        demand_TM=demand_TM,
        J_i_list=J_i_list,
        distIJ=distIJ,
        Q_cap=float(Q),
        T=int(T),
        N=int(N),
        cumulative_install=cumulative_install,
        pre_sorted_J_i=pre_sorted_J_i,
    ))
    proxy_best = float(proxy_curr)

    # FULL score (aligned truth metric)
    ret0 = full_eval_from_U(
        U_curr, m_template, inst, distIJ, policy,
        demand_TM=demand_TM,
        cumulative_install=cumulative_install
    )

    score0 = float(ret0[0])
    _m0_full = ret0[1]
    # cov optional
    _cov0 = ret0[2] if len(ret0) >= 3 else None

    score_curr = float(score0)
    best_full_score = float(score0)
    U_best_full = dict(U_curr)
    m_best_full = _m0_full

    # -------------------------------------------------
    # Logs
    # -------------------------------------------------
    logger = DRLogger()
    dr_trace = []
    batch_logs = []

    logger.log(0, float(proxy_curr), float(best_full_score), 0.0, 0, "init", 1, 1)
    dr_trace.append({
        "iteration": 0,
        "time": 0.0,
        "mode": "init",
        "k_remove": 0,
        "current": float(proxy_curr),
        "after_destroy": np.nan,
        "score_try": np.nan,
        "proxy_try": np.nan,
        "proxy_curr": float(proxy_curr),
        "proxy_best": float(proxy_best),
        "best_full": float(best_full_score),
        "accepted": True,
        "improved_best": True,
        "improved_curr": True,
    })

    seen = set()
    def hash_u(Ud):
        return tuple(Ud[k] for k in sorted(Ud.keys()))
    seen.add(hash_u(U_curr))

    t_start = time.perf_counter()

    # Profiling
    t_destroy = t_reconstruct = t_eval = t_log = 0.0
    n_destroy_calls = n_reconstruct_calls = n_eval_calls = 0

    # Adaptive destroy init
    if adaptive_destroy:
        modes = list(destroy_modes)
        K = len(modes)
        w = np.ones(K, dtype=float)
        p = w / w.sum()

        window_picked = {m: 0 for m in modes}
        window_accepted = {m: 0 for m in modes}
        window_best_delta = {m: 0.0 for m in modes}
        window_reward_sum = {m: 0.0 for m in modes}
        batch_start_best = float(best_full_score)

        best_impr_sum = {m: 0.0 for m in modes}
        best_impr_cnt = {m: 0 for m in modes}
    else:
        modes = [str(destroy_mode)]
        w = None
        p = None
        window_picked = {modes[0]: 0}
        window_accepted = {modes[0]: 0}
        window_best_delta = {modes[0]: 0.0}
        window_reward_sum = {modes[0]: 0.0}
        batch_start_best = float(best_full_score)
        best_impr_sum = {modes[0]: 0.0}
        best_impr_cnt = {modes[0]: 0}

    op_used = {m: 0 for m in modes}
    op_accepted = {m: 0 for m in modes}
    op_best = {m: 0 for m in modes}

    # ------------------------------
    # MAIN LOOP + stagnation kick
    # ------------------------------
    no_improve_batches = 0
    force_iters_left = 0
    forced_mode = None
    no_full_improve_iters = 0

    frac_remove = float(frac_remove)
    accept_epsilon_base = float(accept_epsilon)

    it = 0
    while it < int(max_iter) and (time.perf_counter() - t_start) < float(dr_time_limit):
        it += 1
        seed_iter = None if seed is None else (int(seed) + it)

        # choose mode (forced or adaptive)
        if force_iters_left > 0 and forced_mode is not None:
            mode = str(forced_mode)
            force_iters_left -= 1
        else:
            mode = str(rng.choice(modes, p=p)) if adaptive_destroy else modes[0]

        op_used[mode] += 1
        window_picked[mode] += 1

        # destroy
        _t0 = time.perf_counter()
        U_try, k_removed = destroy_multi_u(
            U_curr,
            inst=inst,
            rng=rng,
            P_T=P_T,
            frac_remove=frac_remove,
            mode=mode,
            seed=seed_iter,
            site_cap=max_chargers_per_site,
            cumulative_install=cumulative_install,
        )
        t_destroy += (time.perf_counter() - _t0)
        n_destroy_calls += 1

        # proxy right after destroy (gives the "dip" in the old plot)
        proxy_after_destroy = float(evaluate_u_numpy_greedy_binary(
            U_try,
            demand_TM=demand_TM,
            J_i_list=J_i_list,
            distIJ=distIJ,
            Q_cap=float(Q),
            T=int(T),
            N=int(N),
            cumulative_install=cumulative_install,
            pre_sorted_J_i=pre_sorted_J_i,
        ))

        # reconstruct (trials) -> pick best CONSISTENT proxy
        _t0 = time.perf_counter()
        best_proxy = -1e18
        best_U_candidate = None

        for _ in range(int(reconstruct_trials)):
            U_fill, _sig = reconstruct_u_dict_fast(
                U_partial=dict(U_try),
                demand_IT=demand_TM,           # IMPORTANT: (T,M)
                P_T=P_T,
                Ij_int=Ij_int,
                U_cap=U_cap,
                Q=float(Q),
                rng=rng,
                cumulative_install=cumulative_install,
                top_k_choice=int(top_k_choice),
            )

            proxy = float(evaluate_u_numpy_greedy_binary(
                U_fill,
                demand_TM=demand_TM,
                J_i_list=J_i_list,
                distIJ=distIJ,
                Q_cap=float(Q),
                T=int(T),
                N=int(N),
                cumulative_install=cumulative_install,
                pre_sorted_J_i=pre_sorted_J_i,
            ))

            if proxy > best_proxy:
                best_proxy = float(proxy)
                best_U_candidate = U_fill

        t_reconstruct += (time.perf_counter() - _t0)
        n_reconstruct_calls += 1

        if best_U_candidate is None:
            best_U_candidate = dict(U_try)

        proxy_try = float(best_proxy)

        # acceptance on PROXY (fast)
        accepted = (proxy_try >= proxy_curr - float(accept_epsilon))
        improved_curr = (proxy_try > proxy_curr + 1e-9)
        proxy_improved_best = (proxy_try > proxy_best + 1e-9)

        reward = 0.0
        did_full_eval = False
        full_score_try = np.nan
        improved_best_full = False
        delta_best_full = 0.0

        if accepted:
            proxy_curr = float(proxy_try)
            U_curr = dict(best_U_candidate)
            op_accepted[mode] += 1
            window_accepted[mode] += 1

            # confirm proxy-best with FULL evaluation
            FULL_EVAL_EVERY = 50        # <-- benchmarking default (try 50 or 100)
            do_full = proxy_improved_best or (it % FULL_EVAL_EVERY == 0)

            if do_full:
                # only update proxy_best when it truly improves
                if proxy_improved_best:
                    proxy_best = float(proxy_try)

                _t1 = time.perf_counter()
                ret_full = full_eval_from_U(
                    best_U_candidate, m_template, inst, distIJ, policy,
                    demand_TM=demand_TM,
                    cumulative_install=cumulative_install
                )

                # supports (score, model) OR (score, model, cov)
                full_score_try = float(ret_full[0])
                _m_tmp = ret_full[1]
                _cov = ret_full[2] if len(ret_full) >= 3 else None

                t_eval += (time.perf_counter() - _t1)
                n_eval_calls += 1
                did_full_eval = True

                if full_score_try > best_full_score + 1e-9:
                    improved_best_full = True
                    delta_best_full = float(full_score_try - best_full_score)
                    best_full_score = float(full_score_try)
                    U_best_full = dict(best_U_candidate)
                    m_best_full = _m_tmp

                    op_best[mode] += 1
                    best_impr_sum[mode] += delta_best_full
                    best_impr_cnt[mode] += 1
                    window_best_delta[mode] += delta_best_full

            # reward (stays the same)
            if improved_best_full:
                reward = float(score_best_w)
            elif improved_curr:
                reward = float(score_improve_w)
            else:
                reward = float(score_accept_w)

            window_reward_sum[mode] += float(reward)
        
        
        # ==========================
        # Track full stagnation
        # ==========================
        if improved_best_full:
            no_full_improve_iters = 0
        else:
            no_full_improve_iters += 1
        # ==========================================
        # ILS STRONG PERTURBATION
        # ==========================================
        if no_full_improve_iters >= 100:

            # Large destruction
            mode_jump = rng.choice(
                ["area_destroy", "local_remove", "site_swap"],
                p=[0.3, 0.2, 0.5]
            )

            U_jump, _ = destroy_multi_u(
                U_curr,
                inst=inst,
                rng=rng,
                P_T=P_T,
                frac_remove=frac_remove_jump,
                mode=mode_jump,              # ✅ correct keyword
                seed=None,
                site_cap=max_chargers_per_site,
                cumulative_install=cumulative_install,
            )

            # Reconstruct once
            # ---------------------------------
            top_k_choice_jump = 20
            reconstruct_trials_jump = 15

            best_local_fast = -1e30
            U_candidate_jump = None
            for _ in range(reconstruct_trials_jump):
                U_fill, fast_proxy = reconstruct_u_dict_fast(
                    U_partial=dict(U_jump),
                    demand_IT=demand_TM,
                    P_T=P_T,
                    Ij_int=Ij_int,
                    U_cap=U_cap,
                    Q=Q_cap,
                    rng=rng,
                    cumulative_install=cumulative_install,
                    top_k_choice=top_k_choice_jump,
                )

                if fast_proxy > best_local_fast:
                    best_local_fast = fast_proxy
                    U_candidate_jump = U_fill

            if U_candidate_jump is not None:
                U_curr = dict(U_candidate_jump)

                proxy_curr = evaluate_u_numpy_greedy_binary(
                    U_curr,
                    demand_TM=demand_TM,
                    J_i_list=J_i_list,
                    distIJ=distIJ,
                    Q_cap=Q_cap,
                    T=T,
                    N=N,
                    cumulative_install=cumulative_install,
                )

            no_full_improve_iters = 0

        
        # trace
        elapsed = float(time.perf_counter() - t_start)
        dr_trace.append({
            "iteration": int(it),
            "time": float(elapsed),
            "mode": str(mode),
            "k_remove": float(k_removed) if k_removed is not None else np.nan,

            # plot layers (proxy)
            "current": float(proxy_curr),
            "after_destroy": float(proxy_after_destroy),
            "score_try": float(proxy_try),

            "proxy_try": float(proxy_try),
            "proxy_curr": float(proxy_curr),
            "proxy_best": float(proxy_best),

            "did_full_eval": bool(did_full_eval),
            "score_try_full": float(full_score_try) if did_full_eval else np.nan,
            "best_full": float(best_full_score),

            "accepted": bool(accepted),
            "improved_curr": bool(improved_curr),
            "proxy_improved_best": bool(proxy_improved_best),
            "improved_best_full": bool(improved_best_full),
        })

        # logger
        _tlog0 = time.perf_counter()
        seen.add(hash_u(U_curr))
        logger.log(
            int(it),
            float(proxy_curr),
            float(best_full_score),
            float(elapsed),
            int(k_removed) if k_removed is not None else 0,
            str(mode),
            int(accepted),
            int(len(seen)),
        )
        t_log += (time.perf_counter() - _tlog0)

        # batch update + kick (use FULL best)
        if adaptive_destroy and (it % int(update_every) == 0):
            batch_end_best = float(best_full_score)
            batch_impr = batch_end_best - float(batch_start_best)

            no_improve_batches = (no_improve_batches + 1) if (batch_impr <= 1e-12) else 0

            if no_improve_batches >= 2:
                frac_remove = min(0.80, float(frac_remove) + 0.15)
                accept_epsilon = min(0.30, float(accept_epsilon) + 0.08)
                w = 0.5 * w + 0.5 * np.mean(w)
                w = np.maximum(w, 1e-9)
                p = w / w.sum()
                forced_mode = "area_destroy"
                force_iters_left = 15
            else:
                frac_remove= max(frac_remove, float(frac_remove) - 0.05)
                accept_epsilon = max(accept_epsilon_base, float(accept_epsilon) - 0.03)
                if force_iters_left <= 0:
                    forced_mode = None

            row = {
                "iter_from": int(it - update_every + 1),
                "iter_to": int(it),
                "best_start": float(batch_start_best),
                "best_end": float(batch_end_best),
                "batch_best_improvement": float(batch_impr),
            }
            for m in modes:
                picked = int(window_picked[m])
                acc = int(window_accepted[m])
                row[f"{m}_picked"] = picked
                row[f"{m}_accepted"] = acc
                row[f"{m}_acc_rate"] = (acc / picked) if picked > 0 else 0.0
                row[f"{m}_best_delta"] = float(window_best_delta[m])
                row[f"{m}_avg_reward"] = (float(window_reward_sum[m]) / picked) if picked > 0 else 0.0

            batch_logs.append(row)

            # ALNS update
            for k_idx, m in enumerate(modes):
                picked = int(window_picked[m])
                avg_reward = (float(window_reward_sum[m]) / picked) if picked > 0 else 0.0
                w[k_idx] = (1.0 - float(reaction)) * float(w[k_idx]) + float(reaction) * float(avg_reward)
            w = np.maximum(w, 1e-9)
            p = w / w.sum()

            batch_start_best = float(best_full_score)
            for m in modes:
                window_picked[m] = 0
                window_accepted[m] = 0
                window_best_delta[m] = 0.0
                window_reward_sum[m] = 0.0

    # ------------------------------
    # End loop: profiling
    # ------------------------------
    total_elapsed = float(time.perf_counter() - t_start)
    t_other = max(0.0, total_elapsed - (t_destroy + t_reconstruct + t_eval + t_log))

    profiling = dict(
        total_elapsed=float(total_elapsed),
        t_destroy=float(t_destroy),
        t_reconstruct=float(t_reconstruct),
        t_eval=float(t_eval),
        t_log=float(t_log),
        t_other=float(t_other),
        n_destroy_calls=int(n_destroy_calls),
        n_reconstruct_calls=int(n_reconstruct_calls),
        n_eval_calls=int(n_eval_calls),
    )

    # contribution summary
    contrib_rows = []
    for m in modes:
        contrib_rows.append({
            "mode": m,
            "best_impr_sum": float(best_impr_sum.get(m, 0.0)),
            "best_impr_cnt": int(best_impr_cnt.get(m, 0)),
            "avg_best_impr": float(best_impr_sum.get(m, 0.0)) / max(1, int(best_impr_cnt.get(m, 0))),
            "used": int(op_used.get(m, 0)),
            "accepted": int(op_accepted.get(m, 0)),
            "best_hits": int(op_best.get(m, 0)),
            
        })
    df_contrib = pd.DataFrame(contrib_rows).sort_values("best_impr_sum", ascending=False)
    def full_eval_model(m_in):
        m = m_in.clone()
        sync_solution_state(m, cumulative_install=cumulative_install)
        m = reassign_y_greedy_multi(m, distIJ, Ji=None, method_name=policy,
                                    cumulative_install=cumulative_install)
        cov = covered_by_period(m, inst)
        return float(np.sum(cov))
    # ------------------------------
    # Exact (aligned metric: same as DR)

    inst_tag = str(inst.get("meta", {}).get("tag", ""))
    m_exact = base_out.get("m_exact", None)

    exact_score = None
    exact_raw_obj = None

    if m_exact is not None and base_out.get("exact_has_feasible", False):
        exact_score = float(full_eval_model(m_exact))          # aligned metric
        exact_raw_obj = float(value(m_exact.obj))              # debug only

    return dict(
        inst_tag=inst_tag,
        policy=policy,

        # DR best (aligned)
        m_best=m_best_full,
        U_best=dict(U_best_full),
        DR_best_total_eval=float(best_full_score),

        # exact (aligned)
        m_exact=m_exact,
        exact_score=float(exact_score) if exact_score is not None else None,
        exact_raw_obj=float(exact_raw_obj) if exact_raw_obj is not None else None,
        exact_has_feasible=bool(base_out.get("exact_has_feasible", False)),
        time_exact=base_out.get("time_exact", None),
        exact_termination=base_out.get("exact_termination", None),
        exact_gap=base_out.get("exact_gap", None),
        proven_optimal=base_out.get("proven_optimal_exact", False),

        # logs
        DR_log=logger.to_df(),
        DR_trace=pd.DataFrame(dr_trace),
        DR_batches=pd.DataFrame(batch_logs),
        DR_contrib=df_contrib,

        # misc
        distIJ=distIJ,
        destroy_modes=list(modes),
        profiling=profiling,
        DR_total_full_evals=int(n_eval_calls),
        DR_best_proxy=float(proxy_best),
        DR_time=float(total_elapsed),
    )

def covered_by_period(m, inst, y_thr=0.5):
    """
    Returns covered demand per period based on y[i,j,t] assignments.
    Safe when some y vars are uninitialized (skips None values).

    demand_IT can be either:
      - (T, M): demand_IT[t][i]
      - (M, T): demand_IT[i][t]
    """

    demand_raw = inst["demand_IT"]
    demand = np.asarray(demand_raw, dtype=float)

    M = len(inst["coords_I"])

    # infer orientation
    if demand.ndim != 2:
        raise ValueError(f"demand_IT must be 2D, got shape={demand.shape}")

    if demand.shape[1] == M:
        # (T, M)
        T = demand.shape[0]
        get_demand = lambda t, i: float(demand[t, i])
    elif demand.shape[0] == M:
        # (M, T)
        T = demand.shape[1]
        get_demand = lambda t, i: float(demand[i, t])
    else:
        raise ValueError(f"demand_IT shape {demand.shape} incompatible with M={M}")

    # Build arcs-by-i once: arcs_by_i[i] = list of feasible sites j
    arcs_by_i = {i: [] for i in range(M)}
    for (ii, j) in m.Arcs:
        ii = int(ii); j = int(j)
        if 0 <= ii < M:
            arcs_by_i[ii].append(j)

    cov = np.zeros(T, dtype=float)

    for t in range(T):
        covered_t = 0.0
        for i in range(M):
            assigned = False

            # check only feasible j for this i
            for j in arcs_by_i.get(i, []):
                # IMPORTANT: y index order assumed (i, j, t)
                yvar = m.y[int(i), int(j), int(t)]
                yv = yvar.value
                if yv is not None and float(yv) >= y_thr:
                    assigned = True
                    break

            if assigned:
                covered_t += get_demand(t, i)

        cov[t] = covered_t

    return cov


def full_eval_from_U(U_dict, m_template, inst, distIJ, policy, demand_TM, cumulative_install=True):
    import numpy as np

    m = m_template.clone()
    _apply_u_matrix(m, U_dict)
    sync_solution_state(m, cumulative_install=cumulative_install)

    # IMPORTANT: this must be the FIXED version (correct indentation, correct policy behavior)
    m = reassign_y_greedy_multi(
        m, distIJ, Ji=None, method_name=policy, cumulative_install=cumulative_install
    )

    cov = covered_by_period(m, inst)      # uses inst["demand_IT"], safe orientation inside
    score = float(np.sum(cov))
    return float(score), m, cov