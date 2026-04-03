import time

import numpy as np
import pandas as pd
from pyomo.environ import value
from pyomo.opt import TerminationCondition

from evcs.geom import build_arcs
from evcs.model import build_base_model, build_multi_period_model
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


    # =================================================
    # 1) EXACT baseline
    # =================================================
    exact_score_aligned = np.nan
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
        exact_score_aligned= evaluate_solution(m_exact, distIJ, demand_I, method_name=policy)["covered_demand"]
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
            distIJ, in_range, Ji, Ij,
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
            distIJ, in_range, Ji, Ij,
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
        exact_score_aligned=exact_score_aligned,
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
    greedy_variant: str = "ring",
):
    import time
    import numpy as np
    from pyomo.environ import value
    from pyomo.opt import TerminationCondition

    from evcs.model import build_multi_period_model
    from evcs.methods import (
        sync_solution_state,
        reassign_y_greedy_multi,
        evaluate_solution_multi,
        evaluate_policy_objective_multi,
    )

    # -------------------------
    # small helpers
    # -------------------------
    def _num(x):
        try:
            if x is None:
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    def _copy_vals(v_exact, v_start):
        if v_exact is None or v_start is None:
            return
        try:
            for k in v_exact:
                try:
                    vv = value(v_start[k])
                    v_exact[k].value = vv
                except Exception:
                    pass
        except Exception:
            pass

    # -------------------------
    # instance sizes
    # -------------------------
    coords_I, coords_J = inst["coords_I"], inst["coords_J"]
    M = len(coords_I)
    N = len(coords_J)

    # -------------------------
    # demand canonicalization -> demand_TM shape (T, M)
    # -------------------------
    demand_arr = np.asarray(inst["demand_IT"], dtype=float)
    if demand_arr.ndim != 2:
        raise ValueError(f"demand_IT must be 2D, got shape={demand_arr.shape}")

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
            raise ValueError(f"demand_IT shape {demand_arr.shape} incompatible with M={M}, T={T}")

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
    # 1) GREEDY
    # -------------------------
    t0 = time.perf_counter()

    m_g = build_multi_period_model(
        M=M, N=N, T=T,
        in_range=in_range, Ji=Ji, Ij=Ij,
        demand_IT=demand_TM,
        Q=Q, P_T=P_T,
        distIJ=distIJ,
        method_name=policy,
        max_chargers_per_site=max_chargers_per_site,
        cumulative_install=cumulative_install,
    )

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
        seed=int(seed or 0),
        cumulative_install=cumulative_install,
        U=U_cap,
        mode="aggregate_then_fill",
    )

    for (j, t), val in U0.items():
        try:
            m_g.u[j, t].value = int(val)
        except Exception:
            pass

    sync_solution_state(m_g, cumulative_install=cumulative_install)
    m_g = reassign_y_greedy_multi(
        m_g, distIJ, Ji,
        method_name=policy,
        cumulative_install=cumulative_install
    )

    score_greedy = float(evaluate_solution_multi(m_g, demand_TM)["covered_demand"])
    time_greedy = float(time.perf_counter() - t0)

    # -------------------------
    # 2) EXACT
    # -------------------------
    m_exact = None
    exact_score_aligned = None
    exact_incumbent_obj_raw = None
    exact_bound_raw = None
    exact_gap_raw = None
    time_exact = None
    exact_term = None
    proven_optimal_exact = False
    exact_has_feasible = False

    best_bound_f = np.nan

    try:
        m_exact = build_multi_period_model(
            M=M, N=N, T=T,
            in_range=in_range, Ji=Ji, Ij=Ij,
            demand_IT=demand_TM,
            Q=Q, P_T=P_T,
            distIJ=distIJ,
            method_name=policy,
            max_chargers_per_site=max_chargers_per_site,
            cumulative_install=cumulative_install,
        )

        _copy_vals(getattr(m_exact, "u", None), getattr(m_g, "u", None))
        _copy_vals(getattr(m_exact, "x", None), getattr(m_g, "x", None))
        _copy_vals(getattr(m_exact, "z", None), getattr(m_g, "z", None))
        _copy_vals(getattr(m_exact, "y", None), getattr(m_g, "y", None))

        t1 = time.perf_counter()
        res = solve_model(
            m_exact,
            verbose=verbose,
            time_limit=float(exact_time_limit),
            mip_gap=float(exact_mip_gap),
            load_solution=True,
        )
        time_exact = float(time.perf_counter() - t1)

        best_bound_f = _num(getattr(res, "best_objective_bound", None))
        exact_term = getattr(res, "termination_condition", None)
        proven_optimal_exact = (exact_term == TerminationCondition.optimal)

        # Feasibility: solver must have loaded an actual integer solution
        # (u[j,t] is initialized to 0.0, so "not None" is always true — unreliable)
        # Instead check best_feasible_objective AND that at least one u > 0
        try:
            bfo = getattr(res, "best_feasible_objective", None)
            bfo_f = float(bfo) if bfo is not None else None
            if bfo_f is not None:
                exact_has_feasible = any(
                    (m_exact.u[j, t].value or 0) > 0.5
                    for j in m_exact.J for t in m_exact.T
                )
            else:
                exact_has_feasible = False
        except Exception:
            exact_has_feasible = False

        if exact_has_feasible:
            # Sync z from u/x, then greedily reassign y — same pattern as greedy/DR evaluation
            sync_solution_state(m_exact, cumulative_install=cumulative_install)
            reassign_y_greedy_multi(
                m_exact, distIJ, Ji=Ji,
                method_name=policy,
                cumulative_install=cumulative_install,
            )

            # aligned metric = covered demand only
            exact_score_aligned = float(
                evaluate_solution_multi(m_exact, demand_TM)["covered_demand"]
            )

            # raw metric = same objective as the MIP policy objective
            exact_incumbent_obj_raw = float(
                evaluate_policy_objective_multi(
                    m_exact, demand_TM, distIJ=distIJ, method_name=policy
                )
            )

        if np.isfinite(best_bound_f):
            exact_bound_raw = float(best_bound_f)

        if (
            exact_incumbent_obj_raw is not None
            and exact_bound_raw is not None
            and abs(exact_incumbent_obj_raw) > 1e-9
        ):
            ratio = abs(exact_bound_raw / exact_incumbent_obj_raw)
            if 0.01 <= ratio <= 100:
                exact_gap_raw = (
                    exact_bound_raw - exact_incumbent_obj_raw
                ) / abs(exact_incumbent_obj_raw)
            else:
                exact_gap_raw = None

    except Exception as e:
        if verbose:
            print("[Exact] failed:", repr(e))
        m_exact = None
        exact_score_aligned = None
        exact_incumbent_obj_raw = None
        exact_bound_raw = None
        exact_gap_raw = None
        time_exact = None
        exact_term = None
        proven_optimal_exact = False
        exact_has_feasible = False

    return dict(
        policy=policy,
        greedy_variant=greedy_variant,

        m_best=m_g,
        m_greedy=m_g,
        score_greedy=score_greedy,
        time_greedy=time_greedy,

        m_exact=m_exact,
        exact_score_aligned=exact_score_aligned,
        exact_incumbent_obj_raw=exact_incumbent_obj_raw,
        exact_bound_raw=exact_bound_raw,
        exact_gap_raw=exact_gap_raw,
        exact_has_feasible=bool(exact_has_feasible),
        exact_termination=exact_term,
        proven_optimal_exact=bool(proven_optimal_exact),
        time_exact=time_exact,

        distIJ=distIJ,
        in_range=in_range,
        Ji=Ji,
        Ij=Ij,
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
        return float(demand_IT[t, i])

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
                        s += float(demand_IT[t, ii])

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

            reach.sort(key=lambda ii: float(demand_IT[t, ii]), reverse=True)

            for ii in reach:
                di = float(demand_IT[t, ii])
                if used + di <= Q + 1e-9:
                    uncovered[ii] = False
                    used += di

            # NOTE: this is *not* the DR proxy metric; only a local signal.
            proxy_total += used

    return U, float(proxy_total)

def _u_to_capacity_array(U_dict, T, N, Q_cap, cumulative_install=True):
    """
    Convert U_dict[(j,t)] = chargers installed at period t at site j
    into cap[t,j] = demand capacity (chargers * Q_cap), optionally cumulative over time.
    """
    cap = np.zeros((T, N), dtype=float)
    for (j, t), val in U_dict.items():
        j = int(j)
        t = int(t)
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
    
    cap = _u_to_capacity_array(U_dict, T=T, N=N, Q_cap=Q_cap, cumulative_install=cumulative_install)

    covered = 0.0
    M = len(J_i_list)
    is_dict = isinstance(distIJ, dict)

    for t in range(T):
        cap_t = cap[t].copy()

        for i in range(M):
            d = float(demand_IT[t, i])
            if d <= 1e-12:
                continue

            js = J_i_list[i]
            if not js:
                continue

            if pre_sorted_J_i is not None:
                js_sorted = pre_sorted_J_i[i]
            else:
                js_sorted = sorted(js, key=lambda j: distIJ[(i, j)] if is_dict else distIJ[i, j])

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


def evaluate_u_numpy_greedy_jt(
    U_dict,
    demand_MT,
    pre_sorted_J_i,
    Q_cap,
    T,
    N,
    cumulative_install=True,
):
    """
    Greedy multi-source allocation with capacities.
    Uses U_dict keys (j, t) and demand_MT shape (M, T).
    Returns total covered demand (float).
    """
    cap = _u_to_capacity_array(U_dict, T=T, N=N, Q_cap=Q_cap, cumulative_install=cumulative_install)

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
    demand_TM,
    J_i_list,
    distIJ,
    Q_cap,
    T,
    N,
    cumulative_install=True,
    pre_sorted_J_i=None,
):
    """
    Greedy evaluation that matches a binary assignment interpretation:
    each (i,t) is either fully covered by one site or not covered.
    No demand splitting across multiple sites.

    Expected demand orientation: demand_TM[t][i] (shape (T, M)).
    """
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

            for j in js_sorted:
                if cap_t[j] + 1e-12 >= d:
                    cap_t[j] -= d
                    covered += d
                    break

    return float(covered)
def local_search_u_proxy(
    U_start,
    inst,
    rng,
    P_T,
    demand_TM,
    J_i_list,
    distIJ,
    Q_cap,
    T,
    N,
    cumulative_install,
    max_chargers_per_site,
    ls_moves,
    ls_frac_remove,
    ls_modes,
    Ij_int=None,              # 🔥 ADD
    top_k_choice=3            # 🔥 ADD
):
    """
    Proxy-guided local search on U-dict only.
    This is separate from DR destroy/reconstruct.
    """
    U_best = dict(U_start)

    proxy_best = float(evaluate_u_numpy_greedy_binary(
        U_best,
        demand_TM=demand_TM,
        J_i_list=J_i_list,
        distIJ=distIJ,
        Q_cap=float(Q_cap),
        T=int(T),
        N=int(N),
        cumulative_install=cumulative_install,
        
    ))

    for _ in range(int(ls_moves)):
        mode = str(rng.choice(list(ls_modes)))

        U_try, _ = destroy_multi_u(
            U_best,
            inst=inst,
            rng=rng,
            P_T=P_T,
            frac_remove=float(ls_frac_remove),
            mode=mode,
            seed=None,
            site_cap=max_chargers_per_site,
            cumulative_install=cumulative_install,
        )

        U_fill, _ = reconstruct_u_dict_fast(
            U_partial=dict(U_try),
            demand_IT=demand_TM,
            P_T=P_T,
            Ij_int=Ij_int,
            U_cap=int(max_chargers_per_site) if max_chargers_per_site is not None else int(max(P_T)),
            Q=float(Q_cap),
            rng=rng,
            cumulative_install=cumulative_install,
            top_k_choice=int(top_k_choice),
        )

        proxy_try = float(evaluate_u_numpy_greedy_binary(
            U_fill,
            demand_TM=demand_TM,
            J_i_list=J_i_list,
            distIJ=distIJ,
            Q_cap=float(Q_cap),
            T=int(T),
            N=int(N),
            cumulative_install=cumulative_install,
           
        ))

        if proxy_try > proxy_best + 1e-9:
            U_best = dict(U_fill)
            proxy_best = float(proxy_try)

    return U_best, float(proxy_best)

def run_DR_multi(
    inst,
    policy: str,
    P_T,
    Q: float,
    D: float,
    T: int,
    max_iter: int = 20,
    dr_time_limit: float = 120.0,
    frac_remove: float = 0.20,
    destroy_mode: str = "local_remove",
    max_chargers_per_site: int | None = None,
    cumulative_install: bool = True,
    seed: int | None = None,
    accept_epsilon: float = 0.0,

    adaptive_destroy: bool = True,
    destroy_modes=["site_swap", "local_remove", "area_destroy"],
    update_every: int = 5,
    reaction: float = 0.25,
    score_best_w: float = 6.0,
    score_improve_w: float = 2.0,
    score_accept_w: float = 0.5,

    top_k_choice: int = 3,

    # NEW algorithm controls
    batch_size: int = 50,
    top_k_full: int = 5,
    ls_moves: int = 8,
    ls_frac_remove: float = 0.08,
    ls_modes=("site_swap", "local_remove"),
):

    rng = np.random.default_rng(seed if seed is not None else 0)

    coords_I, coords_J = inst["coords_I"], inst["coords_J"]
    M = len(coords_I)
    N = len(coords_J)

    # -------------------------
    # Demand (T,M)
    # -------------------------
    demand_TM = np.asarray(inst["demand_IT"], dtype=float)
    if demand_TM.shape != (T, M):
        demand_TM = demand_TM.T

    # -------------------------
    # Arcs
    # -------------------------
    distIJ, in_range, Ji, Ij = build_arcs(coords_I, coords_J, D=D)

    Ij_int = {j: [] for j in range(N)}
    Ji_int = {i: [] for i in range(M)}
    for (i, j) in in_range:
        Ij_int[j].append(i)
        Ji_int[i].append(j)

    J_i_list = [sorted(Ji_int[i], key=lambda j: float(distIJ[i, j])) for i in range(M)]

    # -------------------------
    # Model template
    # -------------------------
    m_template = build_multi_period_model(
        M=M, N=N, T=T,
        in_range=in_range, Ji=Ji, Ij=Ij,
        demand_IT=demand_TM,
        Q=Q, P_T=P_T,
        distIJ=distIJ,
        method_name=policy,
        max_chargers_per_site=max_chargers_per_site,
        cumulative_install=cumulative_install,
    )

    Q_cap = float(m_template.Q.value)

    # -------------------------
    # Initial solution
    # -------------------------
    U_curr = greedy_schedule_multi_from_variants(
        inst, P_T, D, seed=seed
    )

    proxy_curr = evaluate_u_numpy_greedy_binary(
        U_curr, demand_TM, J_i_list, distIJ,
        Q_cap, T, N, cumulative_install
    )

    full0 = full_eval_from_U(
        U_curr, m_template, inst, distIJ, policy,
        demand_TM=demand_TM, cumulative_install=cumulative_install
    )

    best_full_score = float(full0[0])
    U_best = dict(U_curr)

    seen = set()
    seen_max = 2000  # cap: once full, stop deduplication so DR keeps exploring
    def hash_u(U):
        return tuple(sorted(U.items()))

    seen.add(hash_u(U_curr))

    t_start = time.perf_counter()

    # =========================
    # MAIN LOOP (LEFT FLOWCHART)
    # =========================
    trace_records = [] 
    for it in range(max_iter):

        if time.perf_counter() - t_start > dr_time_limit:
            break

        candidate_pool = []

        # -------------------------
        # 1) Generate candidates
        # -------------------------
        for _ in range(batch_size):

            base = U_curr if rng.random() < 0.7 else U_best

            U_partial, _ = destroy_multi_u(
                base, inst, rng, P_T, frac_remove, destroy_mode
            )

            U_cap_per_site = (
                int(max_chargers_per_site)
                if max_chargers_per_site is not None
                else int(max(P_T))          # fallback: no single period can have more than one period's budget
            )

            U_recon, _ = reconstruct_u_dict_fast(
                U_partial, demand_TM, P_T, Ij_int,
                U_cap=U_cap_per_site, Q=Q_cap, rng=rng
            )

            h = hash_u(U_recon)
            if len(seen) < seen_max and h in seen:
                continue
            if len(seen) < seen_max:
                seen.add(h)

            proxy = evaluate_u_numpy_greedy_binary(
                U_recon, demand_TM, J_i_list, distIJ,
                Q_cap, T, N, cumulative_install
            )

            candidate_pool.append((proxy, U_recon))

        if not candidate_pool:
            continue

        # -------------------------
        # 2) Rank by proxy
        # -------------------------
        candidate_pool.sort(reverse=True, key=lambda x: x[0])

        # -------------------------
        # 3) Top-k selection
        # -------------------------
        k = min(top_k_full, len(candidate_pool))
        top_candidates = candidate_pool[:k]

        # -------------------------
        # 4) Local Search (ONLY top-k)
        # -------------------------
        improved_candidates = []

        for proxy, U in top_candidates:
            U_ls, proxy_ls = local_search_u_proxy(
                U, inst, rng, P_T,
                demand_TM, J_i_list, distIJ,
                Q_cap, T, N,
                cumulative_install,
                max_chargers_per_site,
                ls_moves, ls_frac_remove, ls_modes,
                Ij_int=Ij_int,          # 🔥 ADD THIS
                top_k_choice=top_k_choice
            )
            improved_candidates.append((proxy_ls, U_ls))

        # -------------------------
        # 5) Full evaluation
        # -------------------------
        best_batch_score = -1e18
        best_batch_U = None

        for proxy, U in improved_candidates:
            score, _, _ = full_eval_from_U(
                U, m_template, inst, distIJ, policy,
                demand_TM=demand_TM,
                cumulative_install=cumulative_install
            )

            if score > best_batch_score:
                best_batch_score = score
                best_batch_U = U

        # -------------------------
        # 6) Accept / Update
        # -------------------------
        if best_batch_U is not None:

            if best_batch_score >= best_full_score - accept_epsilon:
                U_curr = dict(best_batch_U)
                proxy_curr = proxy

            if best_batch_score > best_full_score:
                best_full_score = best_batch_score
                U_best = dict(best_batch_U)

        # ADD at bottom of the for-loop body:
        trace_records.append({
                "iteration": it,
                "current": float(best_batch_score) if best_batch_U is not None else float(proxy_curr),
                "best_full": float(best_full_score),
            })

    return {
        "U_best": U_best,
        "best_obj": best_full_score,
        "time": time.perf_counter() - t_start,
        "DR_trace": pd.DataFrame(trace_records),
    }



def covered_by_period(m, inst, y_thr=0.5):
  

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