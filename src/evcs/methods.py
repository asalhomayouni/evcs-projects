# src/evcs/methods.py
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple, Optional

import numpy as np
from pyomo.environ import ConstraintList, Objective, maximize, value


# =========================================================
# Model mode detection
# =========================================================
def has_time(m) -> bool:
    return hasattr(m, "T")


def is_multi_charger_model(m) -> bool:
    """
    In your modeling convention:
      - single-period integer model has z[j] and x[j] integer
      - multi-period integer model has z[j,t] and x[j,t] integer
    """
    return hasattr(m, "z")


# =========================================================
# Sync utilities (single + multi period)
# =========================================================
def _safe_int(v) -> int:
    if v is None:
        return 0
    try:
        return int(round(float(v)))
    except Exception:
        return 0


def sync_open_indicator(m):
    """Single-period: z[j] = 1 if x[j] > 0 else 0 (only if z exists)."""
    if not hasattr(m, "z") or not hasattr(m, "x"):
        return
    for j in m.J:
        xj = _safe_int(m.x[j].value)
        m.z[j].value = 1 if xj > 0 else 0


def sync_open_indicator_multi(m):
    """
    Multi-period: z[j,t] = 1 if x[j,t] > 0 else 0.
    No-op if not multi-charger or not time-indexed.
    """
    if not is_multi_charger_model(m) or not has_time(m):
        return
    for t in m.T:
        for j in m.J:
            xjt = _safe_int(m.x[j, t].value)
            m.z[j, t].value = 1 if xjt > 0 else 0

def sync_solution_state(m, cumulative_install: bool = True):
    """
    One call to make a heuristic solution internally consistent.

    Multi-period:
      u -> x (cumulative or not) -> z
    Single-period:
      x -> z   (if multi-charger model)
    """
    if has_time(m):
        # multi-period
        if hasattr(m, "u") and hasattr(m, "x"):
            sync_x_from_u_multi(m, cumulative_install=cumulative_install)

        # z from x (multi-period)
        sync_open_indicator_multi(m)

    else:
        # single-period
        sync_open_indicator(m)

def sync_x_from_u_multi(m, cumulative_install: bool = True):
    """
    Multi-period: build x[j,t] from u[j,t].

    If cumulative_install=True:
        x[j,t] = sum_{k<=t} u[j,k]
    else:
        x[j,t] = u[j,t]

    Also clamps x to [0, U] to avoid Pyomo W1002 warnings.
    """
    if not hasattr(m, "u") or not hasattr(m, "x") or not has_time(m):
        return

    U = None
    if hasattr(m, "U") and m.U is not None:
        try:
            U = int(m.U.value)
        except Exception:
            U = None

    for j in m.J:
        running = 0
        for t in m.T:
            ut = _safe_int(m.u[j, t].value)

            if cumulative_install:
                running += ut
                x_val = running
            else:
                x_val = ut

            # clamp to bounds (0..U) to prevent W1002 warnings
            if U is not None:
                if x_val > U:
                    x_val = U
                elif x_val < 0:
                    x_val = 0
            else:
                if x_val < 0:
                    x_val = 0

            m.x[j, t].value = int(x_val)


# =========================================================
# Unified accessors (single + multi)
# =========================================================
def open_value(m, j) -> float:
    """Single-period open indicator (z preferred, else x)."""
    if is_multi_charger_model(m):
        return float(m.z[j].value or 0.0)
    return float(m.x[j].value or 0.0)


def charger_count(m, j) -> int:
    """Single-period charger count (binary -> 0/1)."""
    return _safe_int(m.x[j].value)


def open_value_t(m, j, t) -> float:
    """Multi-period open indicator at (j,t) (z preferred, else x)."""
    if hasattr(m, "z"):
        return float(m.z[j, t].value or 0.0)
    return float(m.x[j, t].value or 0.0)


def charger_count_t(m, j, t) -> int:
    """Multi-period charger count at (j,t)."""
    return _safe_int(m.x[j, t].value)


def total_chargers(m) -> int:
    """Total chargers in single-period."""
    return sum(charger_count(m, j) for j in m.J)


# =========================================================
# Evaluation (single + multi)
# =========================================================
def evaluate_solution(m, distIJ, demand_I, method_name="closest_only"):
    """
    Evaluate covered demand explicitly (do NOT rely on m.obj potentially being stale).
    """
    total_demand = float(sum(demand_I))
    covered = 0.0
    for (i, j) in m.Arcs:
        covered += float(demand_I[i]) * float(m.y[i, j].value or 0.0)

    cov_pct = 100.0 * covered / total_demand if total_demand > 0 else 0.0
    return {"covered_demand": covered, "covered_pct": cov_pct}


def evaluate_solution_multi(m, demand_IT):
    """
    Multi-period covered demand explicitly:
      sum_{t,i,j} a[i,t] * y[i,j,t]
    demand_IT expected as list: demand_IT[t][i]
    """
    total_demand = 0.0
    total_covered = 0.0

    for t in m.T:
        for i in m.I:
            d = float(demand_IT[int(t)][int(i)])
            total_demand += d

    for (i, j) in m.Arcs:
        for t in m.T:
            d = float(demand_IT[int(t)][int(i)])
            total_covered += d * float(m.y[i, j, t].value or 0.0)

    cov_pct = 100.0 * total_covered / total_demand if total_demand > 0 else 0.0
    return {"covered_demand": total_covered, "covered_pct": cov_pct}


# =========================================================
# Policy helpers
# =========================================================
def compute_farther(distIJ, in_range, Ji):
    """
    farther_of[(i,j)] -> list of arcs (i,k) where k is farther than j for demand i.
    Used by closest_only policy.
    """
    farther_of = {}
    for i, js in Ji.items():
        js_sorted = sorted(list(js), key=lambda j: distIJ[i][j])
        for pos, j in enumerate(js_sorted):
            farther_of[(i, j)] = [(i, k) for k in js_sorted[pos + 1:]]
    return farther_of


def _reset_obj_single(m):
    if hasattr(m, "obj"):
        m.del_component("obj")


def _reset_obj_multi(m):
    if hasattr(m, "obj"):
        m.del_component("obj")


def add_closest_only(m, farther_of):
    m.closest_only = ConstraintList()
    for (i, j), farther in farther_of.items():
        if is_multi_charger_model(m):
            m.closest_only.add(sum(m.y[ii, jj] for (ii, jj) in farther) <= 1 - m.z[j])
        else:
            m.closest_only.add(sum(m.y[ii, jj] for (ii, jj) in farther) <= 1 - m.x[j])
    return m


def add_closest_priority(m, distIJ):
    _reset_obj_single(m)
    expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    expr_tie = sum((1.0 - distIJ[i][j]) * m.y[i, j] for (i, j) in m.Arcs)
    m.obj = Objective(expr=expr_cov + 1e-3 * expr_tie, sense=maximize)
    return m


def add_system_optimum(m, distIJ, lambda_dist=0.1):
    _reset_obj_single(m)
    expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    expr_dist = sum(distIJ[i][j] * m.y[i, j] for (i, j) in m.Arcs)
    m.obj = Objective(expr=expr_cov - lambda_dist * expr_dist, sense=maximize)
    return m


def add_uniform_allocation_constraints(m):
    _reset_obj_single(m)
    expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    m.obj = Objective(expr=expr_cov, sense=maximize)

    m.uniform_alloc = ConstraintList()
    for i in m.I:
        reach = [j for j in m.J if (i, j) in m.Arcs]
        if not reach:
            continue
        frac = 1.0 / len(reach)
        for j in reach:
            if is_multi_charger_model(m):
                m.uniform_alloc.add(m.y[i, j] == frac * m.z[j])
            else:
                m.uniform_alloc.add(m.y[i, j] == frac * m.x[j])
    return m


def apply_method(m, method_name, distIJ, in_range, Ji, Ij, farther_of, verbose=False):
    """
    Single-period policy application (kept for compatibility).
    """
    name = str(method_name).lower()

    if name == "uniform":
        return add_uniform_allocation_constraints(m)

    if name == "closest_only":
        add_closest_only(m, farther_of)
        _reset_obj_single(m)
        expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
        m.obj = Objective(expr=expr_cov, sense=maximize)
        return m

    if name == "closest_priority":
        return add_closest_priority(m, distIJ)

    if name == "system_optimum":
        return add_system_optimum(m, distIJ, lambda_dist=0.1)

    _reset_obj_single(m)
    expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    m.obj = Objective(expr=expr_cov, sense=maximize)
    return m


def apply_method_multi(m, method_name, distIJ, in_range, Ji, Ij, farther_of, verbose=False):
    """
    Multi-period policy constraints replicated per period.
    Falls back to apply_method if model has no T.
    """
    if not has_time(m):
        return apply_method(m, method_name, distIJ, in_range, Ji, Ij, farther_of, verbose=verbose)

    name = str(method_name).lower()

    def set_cov_obj():
        _reset_obj_multi(m)
        expr_cov = sum(m.a[i, t] * m.y[i, j, t] for (i, j) in m.Arcs for t in m.T)
        m.obj = Objective(expr=expr_cov, sense=maximize)

    if name == "uniform":
        _reset_obj_multi(m)
        expr_cov = sum(m.a[i, t] * m.y[i, j, t] for (i, j) in m.Arcs for t in m.T)
        m.obj = Objective(expr=expr_cov, sense=maximize)

        m.uniform_alloc = ConstraintList()
        for t in m.T:
            for i in m.I:
                reach = [j for j in m.J if (i, j) in m.Arcs]
                if not reach:
                    continue
                frac = 1.0 / len(reach)
                for j in reach:
                    if hasattr(m, "z"):
                        m.uniform_alloc.add(m.y[i, j, t] == frac * m.z[j, t])
                    else:
                        m.uniform_alloc.add(m.y[i, j, t] == frac * m.x[j, t])
        return m

    if name == "closest_only":
        # remove old component to avoid Pyomo "Implicitly replacing" warning
        if hasattr(m, "closest_only"):
            m.del_component(m.closest_only)

        m.closest_only = ConstraintList()

        # choose open indicator (z preferred for integer/multi-charger)
        def open_at(j, t):
            if hasattr(m, "z"):
                return m.z[j, t]
            return m.x[j, t]

        # farther_of: dict keyed by (i,j) -> list of farther arcs [(ii,jj),...]
        # add one constraint per (j,t) per anchor (i,j)
        for t in m.T:
            for (i, j), farther in farther_of.items():
                # IMPORTANT: if farther list is empty, skip
                if not farther:
                    continue

                # sum of "farther" assignments at time t must be 0 if site j is open
                # <= 1 - open => if open=1 then sum <=0; if open=0 then sum<=1 (inactive)
                m.closest_only.add(
                    sum(m.y[ii, jj, t] for (ii, jj) in farther) <= 1 - open_at(j, t)
                )

        set_cov_obj()
        return m

    if name == "closest_priority":
        _reset_obj_multi(m)
        expr_cov = sum(m.a[i, t] * m.y[i, j, t] for (i, j) in m.Arcs for t in m.T)
        expr_tie = sum((1.0 - distIJ[i][j]) * m.y[i, j, t] for (i, j) in m.Arcs for t in m.T)
        m.obj = Objective(expr=expr_cov + 1e-3 * expr_tie, sense=maximize)
        return m

    if name == "system_optimum":
        _reset_obj_multi(m)
        lambda_dist = 0.1
        expr_cov = sum(m.a[i, t] * m.y[i, j, t] for (i, j) in m.Arcs for t in m.T)
        expr_dist = sum(distIJ[i][j] * m.y[i, j, t] for (i, j) in m.Arcs for t in m.T)
        m.obj = Objective(expr=expr_cov - lambda_dist * expr_dist, sense=maximize)
        return m

    set_cov_obj()
    return m


# =========================================================
# Greedy assignment (single + multi)
# =========================================================
def reassign_y_greedy(m, distIJ, Ji, method_name: str):
    """
    Single-period greedy assignment:
      - uniform: split among reachable open sites
      - else: assign to nearest open site with remaining capacity
    Capacity at j: Q * x[j]
    """
    I = sorted({i for i, _ in m.Arcs})
    J = sorted({j for _, j in m.Arcs})
    Q = float(m.Q.value)
    a = {i: float(m.a[i]) for i in I}

    # Clear y
    for (ii, jj) in m.Arcs:
        m.y[ii, jj].value = 0.0

    # Remaining capacity
    cap_rem = {j: Q * float(charger_count(m, j)) for j in J}
    open_sites = {j for j in J if open_value(m, j) > 0.5}

    for i in I:
        reachable = list(Ji.get(i, []))
        if not reachable:
            continue

        open_reach = [j for j in reachable if j in open_sites]
        if not open_reach:
            continue

        if method_name == "uniform":
            # Binary-uniform: pick one feasible open site (random)
            j = random.choice(open_reach)
            if cap_rem[j] >= a[i] - 1e-9:
                m.y[i, j].value = 1.0
                cap_rem[j] -= a[i]

        else:
            for j in sorted(open_reach, key=lambda jj: distIJ[i][jj]):
                if cap_rem[j] >= a[i] - 1e-9:
                    m.y[i, j].value = 1.0
                    cap_rem[j] -= a[i]
                    break

    sync_solution_state(m)
    return m


import random

def reassign_y_greedy_multi(m, distIJ, Ji, method_name: str, cumulative_install: bool = True):
    """
    Multi-period greedy assignment (BINARY y).
    - y[i,j,t] is set to 0/1 only
    - each demand i at time t assigned to at most one open site
    - capacity: sum_i a[i,t] * y[i,j,t] <= Q * x[j,t]
    
    Strong greedy:
      - assigns i in descending demand order (per period)
      - uses Best-Fit among feasible open sites (fills capacity tightly)
      - uniform remains random among FEASIBLE open sites
    """
    import random
    sync_solution_state(m, cumulative_install=cumulative_install)

    I_list = [int(i) for i in m.I]
    J_list = [int(j) for j in m.J]
    T_list = [int(t) for t in m.T]
    Q = float(m.Q.value)

    # --- Rebuild adjacency from Arcs (int keys) ---
    Ji_int = {}
    for (i, j) in m.Arcs:
        ii, jj = int(i), int(j)
        Ji_int.setdefault(ii, []).append(jj)

    # --- Clear y (binary) ---
    for (i, j) in m.Arcs:
        ii, jj = int(i), int(j)
        for tt in T_list:
            m.y[ii, jj, tt].value = 0

    # --- Assign per period ---
    for tt in T_list:
        # period demand
        a = {ii: float(m.a[ii, tt]) for ii in I_list}

        # remaining capacity at each site for this period
        cap_rem = {jj: Q * float(charger_count_t(m, jj, tt)) for jj in J_list}

        # open sites this period
        open_sites = {jj for jj in J_list if open_value_t(m, jj, tt) > 0.5}

        # STRONG: assign high-demand nodes first
        I_sorted = sorted(I_list, key=lambda ii: a[ii], reverse=True)

        for ii in I_sorted:
            reachable = Ji_int.get(ii, [])
            if not reachable:
                continue

            open_reach = [jj for jj in reachable if jj in open_sites]
            if not open_reach:
                continue

            # only keep feasible by remaining capacity
            feasible = [jj for jj in open_reach if cap_rem[jj] >= a[ii] - 1e-9]
            if not feasible:
                continue

            if method_name == "uniform":
                # random among feasible (still binary)
                chosen = random.choice(feasible)

            else:
                # STRONG: Best-Fit (min remaining capacity after placing ii)
                # tie-break by distance
                chosen = min(
                    feasible,
                    key=lambda jj: (cap_rem[jj] - a[ii], distIJ[ii][jj])
                )

            m.y[ii, chosen, tt].value = 1
            cap_rem[chosen] -= a[ii]

    return m



# =========================================================
# Initial solution helper (kept; minimal cleanup)
# =========================================================
def _build_Ji_from_arcs(m):
    Ji = {}
    for (i, j) in m.Arcs:
        Ji.setdefault(int(i), []).append(int(j))
    return Ji


def build_initial_solution_weighted(model, distIJ, demand_I, method_name="closest_only", weight_mode="W1"):
    """
    Single-period initializer. (Priority-1: keep stable)
    """
    I = sorted({i for i, _ in model.Arcs})
    J = sorted({j for _, j in model.Arcs})
    a = {i: float(model.a[i]) for i in I}
    P = int(value(model.P))

    # reset
    for j in J:
        if has_time(model):
            # not intended for multi-period initializer
            pass
        else:
            model.x[j].value = 0
            if is_multi_charger_model(model):
                model.z[j].value = 0
    for (i, j) in model.Arcs:
        model.y[i, j].value = 0.0

    W1 = {j: sum(a[i] for i in I if (i, j) in model.Arcs) for j in J}

    if not is_multi_charger_model(model):
        k = min(P, len(J))
        if weight_mode == "W1":
            total = sum(W1.values())
            if total <= 1e-12:
                chosen = random.sample(J, k=k)
            else:
                probs = [W1[j] / total for j in J]
                chosen = list(np.random.choice(J, size=k, replace=False, p=probs))
        elif weight_mode == "W2":
            chosen = []
            remaining = set(J)
            D_cluster = 1.0
            try:
                all_d = []
                for u in J:
                    for v in J:
                        if u != v:
                            all_d.append(distIJ[u][v])
                D_cluster = float(np.median(all_d)) * 0.5 if all_d else 1.0
            except Exception:
                pass

            for _ in range(k):
                pool = list(remaining)
                W2 = {}
                for j in pool:
                    near_cnt = sum(1 for c in chosen if distIJ[j][c] <= D_cluster)
                    W2[j] = W1[j] / (1 + near_cnt)
                total = sum(W2.values())
                pick = random.choice(pool) if total <= 1e-12 else np.random.choice(pool, p=[W2[j]/total for j in pool])
                chosen.append(pick)
                remaining.remove(pick)
        else:
            raise ValueError("weight_mode must be 'W1' or 'W2'")

        for j in chosen:
            model.x[j].value = 1
        reassign_y_greedy(model, distIJ, Ji=_build_Ji_from_arcs(model), method_name=method_name)
        return model

    # multi-charger single-period
    counts = {j: 0 for j in J}
    if weight_mode == "W1":
        weights = np.array([max(1e-12, W1[j]) for j in J], dtype=float)
        probs = weights / weights.sum()
        draw = np.random.multinomial(P, probs)
        for idx, j in enumerate(J):
            counts[j] = int(draw[idx])
    elif weight_mode == "W2":
        for _ in range(P):
            weights = np.array([max(1e-12, W1[j]) / (1.0 + counts[j]) for j in J], dtype=float)
            probs = weights / weights.sum()
            pick = int(np.random.choice(J, p=probs))
            counts[pick] += 1
    else:
        raise ValueError("weight_mode must be 'W1' or 'W2'")

    for j in J:
        model.x[j].value = int(counts[j])
    sync_solution_state(model)

    reassign_y_greedy(model, distIJ, Ji=_build_Ji_from_arcs(model), method_name=method_name)
    return model


# =========================================================
# Local search (single-period only; kept stable for now)
# =========================================================
def local_search(model, distIJ, in_range, Ji, Ij, farther_of,
                 method_name="closest_only", max_iter=100,
                 improvement_rule="first", try_order="seq",
                 logger=None):
    """
    Priority-1: keep stable; just ensure sync is called consistently.
    """
    I = sorted({i for i, _ in model.Arcs})
    J = sorted({j for _, j in model.Arcs})

    def objective(m):
        # objective may be policy modified; safe to read
        return float(value(m.obj))

    def get_order(seq):
        seq = list(seq)
        if try_order == "random":
            random.shuffle(seq)
        return seq

    reassign_y_greedy(model, distIJ, Ji, method_name)
    best_score = objective(model)

    # -----------------------------
    # Multi-charger LS (single-period)
    # -----------------------------
    if is_multi_charger_model(model) and not has_time(model):
        U = int(value(model.U)) if hasattr(model, "U") else 10

        for _ in range(max_iter):
            improved = False
            donors = [j for j in J if charger_count(model, j) > 0]
            receivers = [j for j in J if charger_count(model, j) < U]
            if not donors or not receivers:
                break

            for j_from in get_order(donors):
                for j_to in get_order(receivers):
                    if j_from == j_to:
                        continue

                    old_from = charger_count(model, j_from)
                    old_to = charger_count(model, j_to)

                    for K in (1, 2):
                        if old_from < K:
                            continue
                        if old_to + K > U:
                            continue

                        model.x[j_from].value = old_from - K
                        model.x[j_to].value = old_to + K
                        sync_solution_state(model)
                        reassign_y_greedy(model, distIJ, Ji, method_name)
                        new_score = objective(model)

                        if new_score > best_score + 1e-6:
                            best_score = new_score
                            improved = True
                            break

                        # revert
                        model.x[j_from].value = old_from
                        model.x[j_to].value = old_to
                        sync_solution_state(model)
                        reassign_y_greedy(model, distIJ, Ji, method_name)

                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        return model

    # -----------------------------
    # Binary LS (single-period)
    # -----------------------------
    neighborhoods = ["open_close", "merge", "shift"]
    current_move = 0
    max_no_improve = 5
    no_improve_count = 0

    for _ in range(max_iter):
        improved = False
        move_type = neighborhoods[current_move]

        open_sites = [j for j in J if (model.x[j].value or 0.0) > 0.5]
        closed_sites = [j for j in J if j not in open_sites]

        if move_type == "open_close":
            if open_sites and closed_sites:
                for jc in get_order(open_sites):
                    for jo in get_order(closed_sites):
                        model.x[jc].value = 0
                        model.x[jo].value = 1
                        reassign_y_greedy(model, distIJ, Ji, method_name)
                        new_score = objective(model)
                        if new_score > best_score + 1e-6:
                            best_score = new_score
                            improved = True
                            break
                        model.x[jc].value = 1
                        model.x[jo].value = 0
                        reassign_y_greedy(model, distIJ, Ji, method_name)
                    if improved:
                        break

        elif move_type == "merge":
            if len(open_sites) > 1:
                for jc in get_order(open_sites):
                    model.x[jc].value = 0
                    reassign_y_greedy(model, distIJ, Ji, method_name)
                    new_score = objective(model)
                    if new_score > best_score + 1e-6:
                        best_score = new_score
                        improved = True
                        break
                    model.x[jc].value = 1
                    reassign_y_greedy(model, distIJ, Ji, method_name)

        elif move_type == "shift":
            if open_sites and closed_sites:
                for jc in get_order(open_sites):
                    for jo in get_order(closed_sites):
                        model.x[jc].value = 0
                        model.x[jo].value = 1
                        reassign_y_greedy(model, distIJ, Ji, method_name)
                        new_score = objective(model)
                        if new_score > best_score + 1e-6:
                            best_score = new_score
                            improved = True
                            break
                        model.x[jc].value = 1
                        model.x[jo].value = 0
                        reassign_y_greedy(model, distIJ, Ji, method_name)
                    if improved:
                        break

        if improved:
            no_improve_count = 0
            current_move = 0
        else:
            no_improve_count += 1
            current_move = (current_move + 1) % len(neighborhoods)
            if no_improve_count >= max_no_improve:
                break

    return model


# =========================================================
# D&R helpers (single-period only for now)
# =========================================================
def list_open_sites(model) -> List[int]:
    if is_multi_charger_model(model) and not has_time(model):
        return [j for j in model.J if charger_count(model, j) > 0]
    return [j for j in model.x if (model.x[j].value or 0.0) > 0.5]


def _euclid(p, q) -> float:
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def compute_station_load(model, demand_I) -> Dict[int, float]:
    load = {}
    for j in model.J:
        if open_value(model, j) > 0.5:
            load[j] = 0.0

    for (i, j) in model.Arcs:
        if j in load:
            load[j] += float(demand_I[i]) * float(model.y[i, j].value or 0.0)

    return load


def destroy_partial(
    model,
    k_remove: int,
    mode: str = "random",
    coords_J: Optional[List[Tuple[float, float]]] = None,
    demand_I: Optional[List[float]] = None,
    radius: Optional[float] = None,
    seed: Optional[int] = None,
):
    """
    Single-period destroy.
    Binary: remove k stations.
    Multi-charger: remove k chargers (decrement x across sites).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    mode = (mode or "random").lower()

    if not (is_multi_charger_model(model) and not has_time(model)):
        # binary behavior
        Jopen = list_open_sites(model)
        if not Jopen:
            return model
        k = max(1, min(int(k_remove), len(Jopen)))

        if mode == "random":
            to_remove = list(np.random.choice(Jopen, size=k, replace=False))
        elif mode in ("area", "cluster") and coords_J is not None:
            j0 = random.choice(Jopen)
            center = coords_J[j0]
            dists = sorted(_euclid(center, coords_J[j]) for j in Jopen if j != j0)
            if radius is None:
                radius = dists[len(dists)//2] if dists else 0.0
            in_rad = [j for j in Jopen if _euclid(center, coords_J[j]) <= radius + 1e-12]
            to_remove = sorted(in_rad, key=lambda j: _euclid(center, coords_J[j]))[:k] if len(in_rad) >= k else \
                        sorted(Jopen, key=lambda j: _euclid(center, coords_J[j]))[:k]
        elif mode in ("demand_low", "demand_high") and demand_I is not None:
            load = compute_station_load(model, demand_I)
            pairs = [(j, float(load.get(j, 0.0))) for j in Jopen]
            pairs.sort(key=lambda t: t[1], reverse=(mode == "demand_high"))
            to_remove = [j for (j, _) in pairs[:k]]
        else:
            to_remove = list(np.random.choice(Jopen, size=k, replace=False))

        for j in to_remove:
            model.x[j].value = 0
        for (i, j) in model.Arcs:
            if j in set(to_remove):
                model.y[i, j].value = 0.0
        return model

    # multi-charger single-period: remove k chargers
    total_x = total_chargers(model)
    if total_x <= 0:
        return model
    k = max(1, min(int(k_remove), total_x))

    J = list(model.J)
    xvec = np.array([max(0, charger_count(model, j)) for j in J], dtype=float)
    if xvec.sum() <= 0:
        return model

    probs = xvec / xvec.sum()
    donors = list(np.random.choice(J, size=k, replace=True, p=probs))
    for j in donors:
        cur = charger_count(model, j)
        if cur > 0:
            model.x[j].value = cur - 1

    sync_solution_state(model)

    closed_now = {j for j in model.J if charger_count(model, j) == 0}
    for (i, j) in model.Arcs:
        if j in closed_now:
            model.y[i, j].value = 0.0

    return model


def reconstruction_greedy(model, distIJ, demand_I, D, method_name="closest_only", greedy_mode="deterministic"):
    """
    Single-period reconstruction.
    """
    P = int(value(model.P))

    if not (is_multi_charger_model(model) and not has_time(model)):
        open_now = list_open_sites(model)
        missing = max(0, P - len(open_now))
        if missing <= 0:
            return model

        if greedy_mode == "deterministic":
            J = list(model.x.keys())
            I = range(len(demand_I))
            a = demand_I

            def W1(j):
                return sum(a[i] for i in I if (i, j) in model.Arcs)

            candidates = [j for j in J if j not in set(open_now)]
            for j in sorted(candidates, key=W1, reverse=True)[:missing]:
                model.x[j].value = 1
            return model

        weight_mode = "W1" if greedy_mode == "weighted_W1" else "W2"
        return greedy_add_missing_units_binary(model, distIJ, demand_I, method_name, weight_mode, missing)

    # multi-charger single-period
    cur_total = total_chargers(model)
    missing = max(0, P - cur_total)
    if missing <= 0:
        sync_solution_state(model)
        return model

    I = range(len(demand_I))
    a = demand_I
    J = list(model.J)

    def W1(j):
        return sum(a[i] for i in I if (i, j) in model.Arcs)

    for _ in range(missing):
        if greedy_mode in ("deterministic", "weighted_W1"):
            weights = np.array([max(1e-12, W1(j)) for j in J], dtype=float)
        elif greedy_mode == "weighted_W2":
            weights = np.array([max(1e-12, W1(j)) / (1.0 + charger_count(model, j)) for j in J], dtype=float)
        else:
            weights = np.array([max(1e-12, W1(j)) for j in J], dtype=float)

        probs = weights / weights.sum()
        pick = int(np.random.choice(J, p=probs))
        model.x[pick].value = charger_count(model, pick) + 1

    sync_solution_state(model)
    return model


def greedy_add_missing_units_binary(model, distIJ, demand_I, method_name, weight_mode, k_add):
    J = list(model.x.keys())
    open_now = set(list_open_sites(model))
    remaining = [j for j in J if j not in open_now]
    if not remaining:
        return model

    I = range(len(demand_I))
    a = demand_I

    def W1(j):
        total = 0.0
        for i in I:
            if (i, j) in model.Arcs:
                total += a[i]
        return max(1e-12, total)

    for _ in range(min(int(k_add), len(remaining))):
        weights = {}
        if weight_mode == "W1":
            for j in remaining:
                weights[j] = W1(j)
        else:
            for j in remaining:
                near = sum(1 for c in open_now if distIJ[j][c] <= D)
                weights[j] = W1(j) / (1.0 + near)

        total = sum(weights.values())
        pick = random.choice(remaining) if total <= 1e-12 else np.random.choice(remaining, p=[weights[j]/total for j in remaining])
        model.x[pick].value = 1
        open_now.add(pick)
        remaining.remove(pick)

    return model


# =========================================================
# Side-by-side compare helpers
# =========================================================
def extract_solution(model, demand_I):
    if is_multi_charger_model(model) and not has_time(model):
        open_sites = [j for j in model.J if charger_count(model, j) > 0]
        x_counts = {j: charger_count(model, j) for j in model.J if charger_count(model, j) > 0}
    else:
        open_sites = list_open_sites(model)
        x_counts = {j: int((model.x[j].value or 0.0) > 0.5) for j in open_sites}

    load = compute_station_load(model, demand_I)

    assign = {}
    for (i, j) in model.Arcs:
        if (model.y[i, j].value or 0.0) > 0.5:
            assign[int(i)] = int(j)

    return {"open_sites": open_sites, "x_counts": x_counts, "load": load, "assign": assign}


def compare_solutions(model_A, model_B, demand_I):
    A = extract_solution(model_A, demand_I)
    B = extract_solution(model_B, demand_I)
    setA, setB = set(A["open_sites"]), set(B["open_sites"])
    return {
        "n_open_A": len(setA),
        "n_open_B": len(setB),
        "only_A": sorted(list(setA - setB)),
        "only_B": sorted(list(setB - setA)),
        "common": sorted(list(setA & setB)),
        "x_counts_A": A["x_counts"],
        "x_counts_B": B["x_counts"],
    }