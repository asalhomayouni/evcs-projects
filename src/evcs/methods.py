# src/evcs/methods.py
from __future__ import annotations

import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from pyomo.environ import ConstraintList, Objective, maximize, value


# =========================================================
# Evaluate solution
# =========================================================
def evaluate_solution(m, distIJ, demand_I, method_name="closest_only"):
    """
    Evaluate using the model objective (m.obj) so that exact and heuristic are comparable.
    Returns:
      covered_demand = objective value
      covered_pct    = percent of total demand covered (by y arcs)
    """
    obj_val = float(value(m.obj))
    total_demand = float(sum(demand_I))

    cov_term = sum(float(demand_I[i]) * (m.y[i, j].value or 0.0) for (i, j) in m.Arcs)
    cov_pct = 100.0 * cov_term / total_demand if total_demand > 0 else 0.0

    return {"covered_demand": obj_val, "covered_pct": cov_pct}


# =========================================================
# Policy definitions
# =========================================================
def add_closest_only(m, farther_of):
    m.closest_only = ConstraintList()
    for (i, j), farther in farther_of.items():
        # If j is open, prevent assignment to stations farther than j (policy logic)
        m.closest_only.add(sum(m.y[ii, jj] for (ii, jj) in farther) <= 1 - m.x[j])
    return m


def add_closest_priority(m, distIJ):
    if hasattr(m, "obj"):
        m.del_component("obj")
    expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    expr_tiebreak = sum((1.0 - distIJ[i][j]) * m.y[i, j] for (i, j) in m.Arcs)
    m.add_component("obj", Objective(expr=expr_cov + 1e-3 * expr_tiebreak, sense=maximize))
    return m


def add_system_optimum(m, distIJ, lambda_dist=0.1):
    if hasattr(m, "obj"):
        m.del_component("obj")
    expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    expr_dist = sum(distIJ[i][j] * m.y[i, j] for (i, j) in m.Arcs)
    m.add_component("obj", Objective(expr=expr_cov - lambda_dist * expr_dist, sense=maximize))
    return m


def add_uniform_allocation_constraints(m):
    if hasattr(m, "obj"):
        m.del_component("obj")

    expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    m.add_component("obj", Objective(expr=expr_cov, sense=maximize))

    m.uniform_alloc = ConstraintList()
    for i in m.I:
        reach = [j for j in m.J if (i, j) in m.Arcs]
        if reach:
            frac = 1.0 / len(reach)
            for j in reach:
                m.uniform_alloc.add(m.y[i, j] == frac * m.x[j])
    return m


# =========================================================
# Farther-of helper for closest-only constraints
# =========================================================
def compute_farther(distIJ, in_range, Ji):
    """
    farther_of[(i,j)] -> list of arcs (i,k) where k is farther than j for that i.
    Used by closest_only policy.
    """
    farther_of = {}
    for i, js in Ji.items():
        js_sorted = sorted(list(js), key=lambda j: distIJ[i][j])
        for pos, j in enumerate(js_sorted):
            farther = [(i, k) for k in js_sorted[pos + 1 :]]
            farther_of[(i, j)] = farther
    return farther_of


# =========================================================
# Apply policy logic
# =========================================================
def apply_method(m, method_name, distIJ, in_range, Ji, Ij, farther_of, verbose=False):
    name = str(method_name).lower()

    # If uniform adds constraints, do it first
    if name == "uniform":
        add_uniform_allocation_constraints(m)
        return m

    # Otherwise objective-based variants:
    if name == "closest_only":
        add_closest_only(m, farther_of)
        if hasattr(m, "obj"):
            m.del_component("obj")
        expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
        m.obj = Objective(expr=expr_cov, sense=maximize)
        return m

    if name == "closest_priority":
        return add_closest_priority(m, distIJ)

    if name == "system_optimum":
        return add_system_optimum(m, distIJ, lambda_dist=0.1)

    # Default: coverage objective
    if hasattr(m, "obj"):
        m.del_component("obj")
    expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    m.obj = Objective(expr=expr_cov, sense=maximize)
    return m


# =========================================================
# Greedy Initializers
# =========================================================
def build_initial_solution(model, distIJ, mode="greedy", policy=None):
    """
    Simple greedy baseline. Kept for compatibility.
    """
    I = sorted({i for i, _ in model.Arcs})
    J = sorted({j for _, j in model.Arcs})

    # reset x,y
    for j in J:
        model.x[j].value = 0.0
    for (i, j) in model.Arcs:
        model.y[i, j].value = 0.0

    P = int(value(model.P))
    Q = float(model.Q.value)
    a = {i: float(model.a[i]) for i in I}

    # choose P opens
    if policy == "system_optimum":
        score = {j: sum(distIJ[i][j] for i in I if (i, j) in model.Arcs) for j in J}
        open_list = sorted(score, key=score.get)[: min(P, len(J))]
    elif policy == "closest_priority":
        score = {j: -sum(distIJ[i][j] for i in I if (i, j) in model.Arcs) for j in J}
        open_list = sorted(score, key=score.get, reverse=True)[: min(P, len(J))]
    elif policy == "uniform":
        step = max(1, len(J) // max(1, P))
        open_list = [J[k] for k in range(0, len(J), step)][: min(P, len(J))]
    else:
        open_list = J[: min(P, len(J))]

    for j in open_list:
        model.x[j].value = 1.0

    # assign y with capacity
    cap_rem = {j: Q * (model.x[j].value or 0.0) for j in J}
    for i in I:
        reachable = [j for j in J if (i, j) in model.Arcs]
        open_sites = [j for j in reachable if (model.x[j].value or 0.0) > 0.5]
        if not reachable:
            continue

        if policy == "uniform":
            frac = 1.0 / len(reachable)
            for j in reachable:
                model.y[i, j].value = frac * (model.x[j].value or 0.0)
        else:
            if not open_sites:
                continue
            for j in sorted(open_sites, key=lambda jj: distIJ[i][jj]):
                if cap_rem[j] >= a[i] - 1e-9:
                    model.y[i, j].value = 1.0
                    cap_rem[j] -= a[i]
                    break

    return model


def build_initial_solution_smart(model, distIJ, method_name="closest_only"):
    """
    Deterministic greedy: start empty, open the station with best marginal gain repeatedly.
    """
    I = sorted({i for i, _ in model.Arcs})
    J = sorted({j for _, j in model.Arcs})
    Q = float(model.Q.value)
    a = {i: float(model.a[i]) for i in I}
    P = int(value(model.P))

    def reassign_y(m):
        cap_rem = {j: Q * (m.x[j].value or 0.0) for j in J}
        for (ii, jj) in m.Arcs:
            m.y[ii, jj].value = 0.0

        for i in I:
            reachable = [j for j in J if (i, j) in m.Arcs]
            open_sites = [j for j in reachable if (m.x[j].value or 0.0) > 0.5]
            if not reachable:
                continue

            if method_name == "uniform":
                frac = 1.0 / len(reachable)
                for j in reachable:
                    m.y[i, j].value = frac * (m.x[j].value or 0.0)
            else:
                if not open_sites:
                    continue
                for j in sorted(open_sites, key=lambda jj: distIJ[i][jj]):
                    if cap_rem[j] >= a[i] - 1e-9:
                        m.y[i, j].value = 1.0
                        cap_rem[j] -= a[i]
                        break

    def objective(m):
        return float(value(m.obj))

    for j in J:
        model.x[j].value = 0.0
    for (i, j) in model.Arcs:
        model.y[i, j].value = 0.0

    reassign_y(model)
    current_score = objective(model)

    for _ in range(min(P, len(J))):
        best_gain, best_j = -1e18, None

        closed_sites = [j for j in J if (model.x[j].value or 0.0) < 0.5]
        for j in closed_sites:
            model.x[j].value = 1.0
            reassign_y(model)
            new_score = objective(model)
            gain = new_score - current_score

            # revert
            model.x[j].value = 0.0
            reassign_y(model)

            if gain > best_gain + 1e-9:
                best_gain, best_j = gain, j

        if best_j is None:
            break

        model.x[best_j].value = 1.0
        reassign_y(model)
        current_score += best_gain

    reassign_y(model)
    return model


def build_initial_solution_weighted(model, distIJ, demand_I, method_name="closest_only", weight_mode="W1"):
    """
    Weighted greedy initializer:
      - W1: prefer stations with high demand in range
      - W2: anti-clustering around chosen sites
    """
    I = sorted({i for i, _ in model.Arcs})
    J = sorted({j for _, j in model.Arcs})
    Q = float(model.Q.value)
    a = {i: float(model.a[i]) for i in I}
    P = int(value(model.P))

    # reset x,y
    for j in J:
        model.x[j].value = 0.0
    for (i, j) in model.Arcs:
        model.y[i, j].value = 0.0

    # base W1 score
    W1 = {j: sum(a[i] for i in I if (i, j) in model.Arcs) for j in J}

    chosen = []
    if weight_mode == "W1":
        total = sum(W1.values())
        if total <= 1e-12:
            chosen = random.sample(J, k=min(P, len(J)))
        else:
            probs = [W1[j] / total for j in J]
            chosen = list(np.random.choice(J, size=min(P, len(J)), replace=False, p=probs))

    elif weight_mode == "W2":
        remaining = set(J)
        # compute a clustering radius from J-J distances (rough)
        try:
            all_d = []
            for u in J:
                for v in J:
                    if u != v:
                        all_d.append(distIJ[u][v])
            D_cluster = float(np.median(all_d)) * 0.5 if all_d else 1.0
        except Exception:
            D_cluster = 1.0

        for _ in range(min(P, len(J))):
            W2 = {}
            for j in remaining:
                near_cnt = sum(1 for c in chosen if distIJ[j][c] <= D_cluster)
                W2[j] = W1[j] / (1 + near_cnt)

            total = sum(W2.values())
            pool = list(remaining)
            if total <= 1e-12:
                pick = random.choice(pool)
            else:
                probs = [W2[j] / total for j in pool]
                pick = np.random.choice(pool, p=probs)

            chosen.append(pick)
            remaining.remove(pick)
    else:
        raise ValueError("weight_mode must be 'W1' or 'W2'")

    # open chosen
    for j in chosen:
        model.x[j].value = 1.0

    # assign y with capacity
    cap_rem = {j: Q * (model.x[j].value or 0.0) for j in J}
    for i in I:
        reach = [j for j in J if (i, j) in model.Arcs]
        open_sites = [j for j in reach if (model.x[j].value or 0.0) > 0.5]
        if not reach or not open_sites:
            continue
        for j in sorted(open_sites, key=lambda jj: distIJ[i][jj]):
            if cap_rem[j] >= a[i] - 1e-9:
                model.y[i, j].value = 1.0
                cap_rem[j] -= a[i]
                break

    return model


# =========================================================
# Local Search (kept close to your original)
# =========================================================
def local_search(model, distIJ, in_range, Ji, Ij, farther_of,
                 method_name="closest_only", max_iter=100,
                 improvement_rule="first", try_order="seq",
                 logger=None):
    """
    Multi-neighborhood local search with y reassignment after each move.
    Neighborhoods: open_close, merge, shift
    """
    I = sorted({i for i, _ in model.Arcs})
    J = sorted({j for _, j in model.Arcs})
    Q = float(model.Q.value)
    a = {i: float(model.a[i]) for i in I}

    def reassign_y(m):
        cap_rem = {j: Q * (m.x[j].value or 0.0) for j in J}
        for (ii, jj) in m.Arcs:
            m.y[ii, jj].value = 0.0

        for i in I:
            reachable = [j for j in J if (i, j) in m.Arcs]
            open_sites = [j for j in reachable if (m.x[j].value or 0.0) > 0.5]
            if not reachable:
                continue

            if method_name == "uniform":
                frac = 1.0 / len(reachable)
                for j in reachable:
                    m.y[i, j].value = frac * (m.x[j].value or 0.0)
            else:
                if not open_sites:
                    continue
                for j in sorted(open_sites, key=lambda jj: distIJ[i][jj]):
                    if cap_rem[j] >= a[i] - 1e-9:
                        m.y[i, j].value = 1.0
                        cap_rem[j] -= a[i]
                        break

    def objective(m):
        return float(value(m.obj))

    def get_order(seq):
        seq = list(seq)
        if try_order == "random":
            random.shuffle(seq)
        return seq

    reassign_y(model)
    best_score = objective(model)

    neighborhoods = ["open_close", "merge", "shift"]
    current_move = 0
    max_no_improve = 5
    no_improve_count = 0

    def apply_best_improvement(candidates, apply_move_fn, revert_fn, it, mv_name):
        nonlocal best_score
        best_gain = 0.0
        best_mv = None

        for data in candidates:
            apply_move_fn(data)
            reassign_y(model)
            new_score = objective(model)
            gain = new_score - best_score

            if gain > best_gain + 1e-6:
                best_gain = gain
                best_mv = data

            revert_fn(data)
            reassign_y(model)

        if best_mv is not None:
            apply_move_fn(best_mv)
            reassign_y(model)
            best_score += best_gain
            return True
        return False

    for it in range(max_iter):
        improved = False
        move_type = neighborhoods[current_move]

        open_sites = [j for j in J if (model.x[j].value or 0.0) > 0.5]
        closed_sites = [j for j in J if j not in open_sites]

        # 1) OPEN-CLOSE
        if move_type == "open_close":
            if not open_sites or not closed_sites:
                improved = False
            elif improvement_rule == "best":
                candidates = [(jc, jo) for jc in get_order(open_sites) for jo in get_order(closed_sites)]

                def apply(data):
                    jc, jo = data
                    model.x[jc].value = 0
                    model.x[jo].value = 1

                def revert(data):
                    jc, jo = data
                    model.x[jc].value = 1
                    model.x[jo].value = 0

                improved = apply_best_improvement(candidates, apply, revert, it, "open_close")
            else:
                for jc in get_order(open_sites):
                    for jo in get_order(closed_sites):
                        model.x[jc].value = 0
                        model.x[jo].value = 1
                        reassign_y(model)
                        new_score = objective(model)
                        if new_score > best_score + 1e-6:
                            best_score = new_score
                            improved = True
                            break
                        # revert
                        model.x[jc].value = 1
                        model.x[jo].value = 0
                        reassign_y(model)
                    if improved:
                        break

        # 2) MERGE (close one open station if redundant; quick heuristic)
        elif move_type == "merge":
            if len(open_sites) <= 1:
                improved = False
            else:
                # try closing a station and see if objective improves after reassignment
                cand = get_order(open_sites)
                for jc in cand:
                    model.x[jc].value = 0
                    reassign_y(model)
                    new_score = objective(model)
                    if new_score > best_score + 1e-6:
                        best_score = new_score
                        improved = True
                        break
                    # revert
                    model.x[jc].value = 1
                    reassign_y(model)

        # 3) SHIFT (move one open station to a nearby closed station)
        elif move_type == "shift":
            if not open_sites or not closed_sites:
                improved = False
            else:
                # pick random open, try some closed
                for jc in get_order(open_sites):
                    for jo in get_order(closed_sites):
                        model.x[jc].value = 0
                        model.x[jo].value = 1
                        reassign_y(model)
                        new_score = objective(model)
                        if new_score > best_score + 1e-6:
                            best_score = new_score
                            improved = True
                            break
                        # revert
                        model.x[jc].value = 1
                        model.x[jo].value = 0
                        reassign_y(model)
                    if improved:
                        break

        # advance neighborhood
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
# D&R helpers (Step 2/3/4)
# =========================================================
def list_open_sites(model) -> List[int]:
    return [j for j in model.x if (model.x[j].value or 0.0) > 0.5]


def _euclid(p, q) -> float:
    return math.hypot(float(p[0]) - float(q[0]), float(p[1]) - float(q[1]))


def compute_station_load(model, demand_I) -> Dict[int, float]:
    """
    load[j] = sum_i demand[i] * y[i,j] (only meaningful after reassign_y)
    """
    load = {}
    for j in model.x:
        if (model.x[j].value or 0.0) > 0.5:
            load[j] = 0.0

    # model.Arcs is (i,j)
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
    Destroy operator supporting:
      - random
      - area     (remove cluster around a random open site)
      - cluster  (same as area; kept for readability)
      - demand_low  (remove lowest-load stations)
      - demand_high (remove highest-load stations)

    IMPORTANT: this function MUTATES the passed model (so caller should clone first).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    Jopen = list_open_sites(model)
    if not Jopen:
        return model

    k = max(1, min(int(k_remove), len(Jopen)))
    mode = (mode or "random").lower()

    # --- decide which stations to remove
    if mode in ("random",):
        to_remove = list(np.random.choice(Jopen, size=k, replace=False))

    elif mode in ("area", "cluster"):
        if coords_J is None:
            # fallback to random
            to_remove = list(np.random.choice(Jopen, size=k, replace=False))
        else:
            j0 = random.choice(Jopen)
            center = coords_J[j0]
            # default radius: based on median distance from j0 to all Jopen
            dists = sorted(_euclid(center, coords_J[j]) for j in Jopen if j != j0)
            if radius is None:
                radius = dists[len(dists)//2] if dists else 0.0

            # candidates in radius (or nearest if too few)
            in_rad = [j for j in Jopen if _euclid(center, coords_J[j]) <= radius + 1e-12]
            if len(in_rad) >= k:
                # remove closest in radius
                in_rad_sorted = sorted(in_rad, key=lambda j: _euclid(center, coords_J[j]))
                to_remove = in_rad_sorted[:k]
            else:
                # remove nearest open sites
                Jopen_sorted = sorted(Jopen, key=lambda j: _euclid(center, coords_J[j]))
                to_remove = Jopen_sorted[:k]

    elif mode in ("demand_low", "demand_high"):
        if demand_I is None:
            to_remove = list(np.random.choice(Jopen, size=k, replace=False))
        else:
            load = compute_station_load(model, demand_I)
            # If some loads missing, set 0
            pairs = [(j, float(load.get(j, 0.0))) for j in Jopen]
            pairs.sort(key=lambda t: t[1], reverse=(mode == "demand_high"))
            to_remove = [j for (j, _) in pairs[:k]]

    else:
        # unknown mode -> random
        to_remove = list(np.random.choice(Jopen, size=k, replace=False))

    # --- apply removal: close x[j], clear all y[i,j]
    for j in to_remove:
        model.x[j].value = 0.0
    for (i, j) in model.Arcs:
        if j in to_remove:
            model.y[i, j].value = 0.0

    return model


def greedy_add_missing_sites(model, distIJ, demand_I, D, method_name, weight_mode, k_add):
    """
    Append-only greedy: add k_add new stations WITHOUT clearing existing ones.
    """
    J = list(model.x.keys())
    open_now = set(list_open_sites(model))
    remaining = [j for j in J if j not in open_now]

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
        elif weight_mode == "W2":
            for j in remaining:
                near = sum(1 for c in open_now if distIJ[j][c] <= D)
                weights[j] = W1(j) / (1.0 + near)
        else:
            for j in remaining:
                weights[j] = W1(j)

        total = sum(weights.values())
        if total <= 1e-12:
            pick = random.choice(remaining)
        else:
            pool = list(remaining)
            probs = [weights[j] / total for j in pool]
            pick = np.random.choice(pool, p=probs)

        model.x[pick].value = 1.0
        open_now.add(pick)
        remaining.remove(pick)

    return model


def reconstruction_greedy(model, distIJ, demand_I, D, method_name="closest_only", greedy_mode="deterministic"):
    """
    Reconstruction after destroy:
      - deterministic: fill missing stations by best W1-like score deterministically
      - weighted_W1 / weighted_W2: probabilistic add
    """
    P = int(value(model.P))
    open_now = list_open_sites(model)
    missing = max(0, P - len(open_now))
    if missing <= 0:
        return model

    if greedy_mode == "deterministic":
        # deterministic: pick top missing by W1 score
        J = list(model.x.keys())
        I = range(len(demand_I))
        a = demand_I

        def W1(j):
            return sum(a[i] for i in I if (i, j) in model.Arcs)

        candidates = [j for j in J if j not in set(open_now)]
        candidates_sorted = sorted(candidates, key=W1, reverse=True)
        to_add = candidates_sorted[:missing]
        for j in to_add:
            model.x[j].value = 1.0
        return model

    elif greedy_mode == "weighted_W1":
        return greedy_add_missing_sites(model, distIJ, demand_I, D, method_name, "W1", missing)

    elif greedy_mode == "weighted_W2":
        return greedy_add_missing_sites(model, distIJ, demand_I, D, method_name, "W2", missing)

    else:
        # fallback
        return greedy_add_missing_sites(model, distIJ, demand_I, D, method_name, "W1", missing)


# =========================================================
# Side-by-side compare helpers (Step 4)
# =========================================================
def extract_solution(model, demand_I):
    open_sites = list_open_sites(model)
    load = compute_station_load(model, demand_I)

    # assignment i -> j (if y[i,j] ~ 1)
    assign = {}
    for (i, j) in model.Arcs:
        if (model.y[i, j].value or 0.0) > 0.5:
            assign[int(i)] = int(j)

    return {"open_sites": open_sites, "load": load, "assign": assign}


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
    }


# =========================================================
# Placeholder SA (kept for compatibility if imported)
# =========================================================
def simulated_annealing(*args, **kwargs):
    raise NotImplementedError("SA not used in the cleaned pipeline.")


def apply_method_placeholder(*args, **kwargs):
    raise NotImplementedError


def compute_farther_placeholder(*args, **kwargs):
    raise NotImplementedError
