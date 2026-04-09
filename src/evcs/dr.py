import time

import numpy as np
import pandas as pd
from pyomo.environ import value
from pyomo.opt import TerminationCondition

from evcs.geom import build_arcs
from evcs.model import build_multi_period_model
from evcs.solve import solve_model
from evcs.methods import (
    sync_solution_state,
    reassign_y_greedy_multi,
    evaluate_solution_multi,
    evaluate_policy_objective_multi,
    greedy_schedule_multi_from_variants,
)
from evcs.proxy import evaluate_u_numpy_greedy_binary
from evcs.destroy import destroy_multi_u
from evcs.greedy import reconstruct_u_dict_fast
from evcs.local_search import local_search_u_proxy
from evcs.full_eval import full_eval_from_U


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
    # Arcs — reuse from inst if available so forbid_self matches exactly
    # -------------------------
    if "in_range" in inst and "distIJ" in inst and "Ji" in inst and "Ij" in inst:
        distIJ  = inst["distIJ"]
        in_range = inst["in_range"]
        Ji      = inst["Ji"]
        Ij      = inst["Ij"]
    else:
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
    eval_id = 0
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

            # log individual proxy evaluation for trace visualisation
            trace_records.append({
                "eval_id":   eval_id,
                "iteration": it,
                "phase":     "dr",
                "proxy":     float(proxy),
                "best_full": float(best_full_score),
            })
            eval_id += 1

        if not candidate_pool:
            continue

        # -------------------------
        # 2) Rank by proxy
        # -------------------------
        candidate_pool.sort(reverse=True, key=lambda x: x[0])\

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

        # record one checkpoint per iteration with updated best_full
        trace_records.append({
            "eval_id":   eval_id,
            "iteration": it,
            "phase":     "checkpoint",
            "proxy":     float(best_batch_score) if best_batch_U is not None else float(proxy_curr),
            "best_full": float(best_full_score),
        })
        eval_id += 1

    return {
        "U_best": U_best,
        "best_obj": best_full_score,
        "time": time.perf_counter() - t_start,
        "DR_trace": pd.DataFrame(trace_records),
    }
