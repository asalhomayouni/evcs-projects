# src/evcs/methods.py
from __future__ import annotations

import random
import math
import numpy as np
from typing import Dict, List, Tuple, Optional

from pyomo.environ import ConstraintList, Objective, maximize, value


# =========================================================
# Helpers: detect integer-charger mode & keep z consistent
# =========================================================
def is_multi_charger_model(m) -> bool:
    return hasattr(m, "z")  # in your new model.py, z exists only when allow_multi_charger=True


def sync_open_indicator(m):
    """
    Keep z[j] consistent with x[j] when allow_multi_charger=True:
      z[j] = 1 if x[j] >= 1 else 0
    No-op for binary case.
    """
    if not is_multi_charger_model(m):
        return
    for j in m.J:
        xj = int(m.x[j].value or 0)
        m.z[j].value = 1 if xj > 0 else 0


def open_value(m, j) -> float:
    """Return an 'open' indicator usable in constraints/logic."""
    if is_multi_charger_model(m):
        return float(m.z[j].value or 0.0)
    return float(m.x[j].value or 0.0)


def charger_count(m, j) -> int:
    """Return integer charger count for both modes (binary -> 0/1)."""
    xj = m.x[j].value
    if xj is None:
        return 0
    return int(round(float(xj)))


def total_chargers(m) -> int:
    return sum(charger_count(m, j) for j in m.J)


# =========================================================
# Evaluate solution
# =========================================================
def evaluate_solution(m, distIJ, demand_I, method_name="closest_only"):
    """
    Evaluate using the model objective (m.obj) so that exact and heuristic are comparable.
    """
    obj_val = float(value(m.obj))
    total_demand = float(sum(demand_I))

    cov_term = sum(float(demand_I[i]) * float(m.y[i, j].value or 0.0) for (i, j) in m.Arcs)
    cov_pct = 100.0 * cov_term / total_demand if total_demand > 0 else 0.0

    return {"covered_demand": obj_val, "covered_pct": cov_pct}


# =========================================================
# Policy definitions (updated for multi-charger using z)
# =========================================================
def add_closest_only(m, farther_of):
    m.closest_only = ConstraintList()
    for (i, j), farther in farther_of.items():
        # if site j is open, prevent assignment to farther arcs
        if is_multi_charger_model(m):
            m.closest_only.add(sum(m.y[ii, jj] for (ii, jj) in farther) <= 1 - m.z[j])
        else:
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
    """
    Uniform allocation only makes sense if there are open sites.
    In multi-charger mode, open indicator is z[j]; otherwise x[j].
    """
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
                if is_multi_charger_model(m):
                    m.uniform_alloc.add(m.y[i, j] == frac * m.z[j])
                else:
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
            farther = [(i, k) for k in js_sorted[pos + 1:]]
            farther_of[(i, j)] = farther
    return farther_of


# =========================================================
# Apply policy logic
# =========================================================
def apply_method(m, method_name, distIJ, in_range, Ji, Ij, farther_of, verbose=False):
    name = str(method_name).lower()

    if name == "uniform":
        add_uniform_allocation_constraints(m)
        return m

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
# Core reassignment used by greedy + LS (updated for integer chargers)
# =========================================================
def reassign_y_greedy(m, distIJ, Ji, method_name: str):
    """
    Greedy assignment:
      - uniform: split among reachable open sites (indicator)
      - else: assign each i to nearest open j with remaining capacity
    Capacity per site j is Q * x[j] (x is integer chargers/modules).
    """
    I = sorted({i for i, _ in m.Arcs})
    J = sorted({j for _, j in m.Arcs})
    Q = float(m.Q.value)
    a = {i: float(m.a[i]) for i in I}

    # clear y
    for (ii, jj) in m.Arcs:
        m.y[ii, jj].value = 0.0

    # capacity remaining
    cap_rem = {j: Q * float(charger_count(m, j)) for j in J}

    # open set (based on z or x)
    open_sites = {j for j in J if open_value(m, j) > 0.5}

    for i in I:
        reachable = list(Ji.get(i, []))
        if not reachable:
            continue

        if method_name == "uniform":
            # distribute only across open sites (if none open -> all zeros)
            open_reach = [j for j in reachable if j in open_sites]
            if not open_reach:
                continue
            frac = 1.0 / len(open_reach)
            for j in open_reach:
                m.y[i, j].value = frac
        else:
            open_reach = [j for j in reachable if j in open_sites]
            if not open_reach:
                continue
            for j in sorted(open_reach, key=lambda jj: distIJ[i][jj]):
                if cap_rem[j] >= a[i] - 1e-9:
                    m.y[i, j].value = 1.0
                    cap_rem[j] -= a[i]
                    break

    sync_open_indicator(m)
    return m


# =========================================================
# Greedy Initializers (updated)
# =========================================================
def build_initial_solution_weighted(model, distIJ, demand_I, method_name="closest_only", weight_mode="W1"):
    """
    Weighted initializer.

    Binary mode:
      - choose P sites (x=1)

    Multi-charger mode:
      - allocate total P chargers across sites (x integer, sum x = P)
      - selection is weighted by W1 or W2; multiple chargers can go to same site
    """
    I = sorted({i for i, _ in model.Arcs})
    J = sorted({j for _, j in model.Arcs})
    a = {i: float(model.a[i]) for i in I}
    P = int(value(model.P))

    # reset
    for j in J:
        model.x[j].value = 0
        if is_multi_charger_model(model):
            model.z[j].value = 0
    for (i, j) in model.Arcs:
        model.y[i, j].value = 0.0

    # base W1 score: total demand in range of j
    W1 = {j: sum(a[i] for i in I if (i, j) in model.Arcs) for j in J}

    if not is_multi_charger_model(model):
        # ---- binary: choose P distinct sites ----
        k = min(P, len(J))
        if weight_mode == "W1":
            total = sum(W1.values())
            if total <= 1e-12:
                chosen = random.sample(J, k=k)
            else:
                probs = [W1[j] / total for j in J]
                chosen = list(np.random.choice(J, size=k, replace=False, p=probs))
        elif weight_mode == "W2":
            # mild anti-clustering: sequential pick
            chosen = []
            remaining = set(J)
            # cluster radius heuristic
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

    # ---- multi-charger: allocate P chargers with replacement (counts) ----
    # Probability proportional to W1 or W2-like score
    counts = {j: 0 for j in J}

    if weight_mode == "W1":
        weights = np.array([max(1e-12, W1[j]) for j in J], dtype=float)
    elif weight_mode == "W2":
        # W2: penalize already chosen sites (soft anti-clustering by count)
        # implement as iterative picks
        for _ in range(P):
            weights = np.array([max(1e-12, W1[j]) / (1.0 + counts[j]) for j in J], dtype=float)
            probs = weights / weights.sum()
            pick = int(np.random.choice(J, p=probs))
            counts[pick] += 1
        # assign to model and finish
        for j in J:
            model.x[j].value = int(counts[j])
        sync_open_indicator(model)
        reassign_y_greedy(model, distIJ, Ji=_build_Ji_from_arcs(model), method_name=method_name)
        return model
    else:
        raise ValueError("weight_mode must be 'W1' or 'W2'")

    # W1 fast allocation (multinomial)
    probs = weights / weights.sum()
    draw = np.random.multinomial(P, probs)
    for idx, j in enumerate(J):
        counts[j] = int(draw[idx])

    for j in J:
        model.x[j].value = int(counts[j])
    sync_open_indicator(model)

    reassign_y_greedy(model, distIJ, Ji=_build_Ji_from_arcs(model), method_name=method_name)
    return model


def _build_Ji_from_arcs(m):
    """Small utility: build Ji dict from m.Arcs when caller doesn't have it."""
    Ji = {}
    for (i, j) in m.Arcs:
        Ji.setdefault(int(i), []).append(int(j))
    return Ji


# =========================================================
# Local Search (updated for integer chargers)
# =========================================================
def local_search(model, distIJ, in_range, Ji, Ij, farther_of,
                 method_name="closest_only", max_iter=100,
                 improvement_rule="first", try_order="seq",
                 logger=None):
    """
    Binary mode neighborhoods:
      - open_close (swap one open with one closed)
      - merge (try closing redundant)
      - shift (swap one open with one closed)

    Multi-charger neighborhoods:
      - shift_charger: move 1 charger from a site with x>0 to another site (x increases)
      - optional relocate: remove all chargers from a site and move them elsewhere (coarse)
    """
    I = sorted({i for i, _ in model.Arcs})
    J = sorted({j for _, j in model.Arcs})

    def objective(m):
        return float(value(m.obj))

    def get_order(seq):
        seq = list(seq)
        if try_order == "random":
            random.shuffle(seq)
        return seq

    # Use shared greedy reassignment (consistent across modes)
    reassign_y_greedy(model, distIJ, Ji, method_name)
    best_score = objective(model)

    # -----------------------------
    # Multi-charger LS
    # -----------------------------
    if is_multi_charger_model(model):
        U = int(value(model.U)) if hasattr(model, "U") else 10

        def open_sites():
            return [j for j in J if charger_count(model, j) > 0]

        def apply_shift(data):
            j_from, j_to = data
            model.x[j_from].value = charger_count(model, j_from) - 1
            model.x[j_to].value = charger_count(model, j_to) + 1
            # enforce bounds
            if charger_count(model, j_to) > U:
                model.x[j_to].value = U
                model.x[j_from].value = charger_count(model, j_from)  # revert count locally

        def revert_shift(data):
            j_from, j_to = data
            model.x[j_from].value = charger_count(model, j_from) + 1
            model.x[j_to].value = charger_count(model, j_to) - 1

        # Main loop: first improvement by default
        # Main loop: first improvement by default
        for it in range(max_iter):
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
                    old_to   = charger_count(model, j_to)

                    for K in (1, 2):
                        if old_from < K:
                            continue
                        if old_to + K > U:
                            continue

                        model.x[j_from].value = old_from - K
                        model.x[j_to].value   = old_to + K
                        sync_open_indicator(model)

                        reassign_y_greedy(model, distIJ, Ji, method_name)
                        new_score = objective(model)

                        if new_score > best_score + 1e-6:
                            best_score = new_score
                            improved = True
                            break

                        # revert this K move
                        model.x[j_from].value = old_from
                        model.x[j_to].value   = old_to
                        sync_open_indicator(model)
                        reassign_y_greedy(model, distIJ, Ji, method_name)

                    if improved:
                        break
                if improved:
                    break

            # ✅ THIS LINE WAS MISSING
            if not improved:
                break

        return model


    # -----------------------------
    # Binary LS (your original style)
    # -----------------------------
    neighborhoods = ["open_close", "merge", "shift"]
    current_move = 0
    max_no_improve = 5
    no_improve_count = 0

    for it in range(max_iter):
        improved = False
        move_type = neighborhoods[current_move]

        open_sites = [j for j in J if (model.x[j].value or 0.0) > 0.5]
        closed_sites = [j for j in J if j not in open_sites]

        if move_type == "open_close":
            if not open_sites or not closed_sites:
                improved = False
            else:
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
                        # revert
                        model.x[jc].value = 1
                        model.x[jo].value = 0
                        reassign_y_greedy(model, distIJ, Ji, method_name)
                    if improved:
                        break

        elif move_type == "merge":
            if len(open_sites) <= 1:
                improved = False
            else:
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
            if not open_sites or not closed_sites:
                improved = False
            else:
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
# D&R helpers (updated for integer chargers)
# =========================================================
def list_open_sites(model) -> List[int]:
    if is_multi_charger_model(model):
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
    Destroy operator.

    Binary mode: remove k stations (set x=0)
    Multi-charger: remove k chargers (decrement x across selected sites), keeping sum x reduced
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    mode = (mode or "random").lower()

    if not is_multi_charger_model(model):
        # -------- binary behavior (your previous logic) --------
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
            if len(in_rad) >= k:
                to_remove = sorted(in_rad, key=lambda j: _euclid(center, coords_J[j]))[:k]
            else:
                to_remove = sorted(Jopen, key=lambda j: _euclid(center, coords_J[j]))[:k]
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

    # -------- multi-charger behavior: remove k chargers --------
    total_x = total_chargers(model)
    if total_x <= 0:
        return model

    k = max(1, min(int(k_remove), total_x))

    # Build list of "charger units" by site (multiset expansion)
    # We won't literally expand (could be large); instead pick sites with probability ~ x[j]
    J = list(model.J)
    xvec = np.array([max(0, charger_count(model, j)) for j in J], dtype=float)
    if xvec.sum() <= 0:
        return model

    # choose donor sites for each removed charger
    probs = xvec / xvec.sum()
    donors = list(np.random.choice(J, size=k, replace=True, p=probs))

    # apply decrements
    for j in donors:
        cur = charger_count(model, j)
        if cur > 0:
            model.x[j].value = cur - 1

    sync_open_indicator(model)

    # clear y columns for sites that became closed (x->0)
    closed_now = {j for j in model.J if charger_count(model, j) == 0}
    for (i, j) in model.Arcs:
        if j in closed_now:
            model.y[i, j].value = 0.0

    return model


def reconstruction_greedy(model, distIJ, demand_I, D, method_name="closest_only", greedy_mode="deterministic"):
    """
    Reconstruction after destroy.

    Binary mode: add missing stations until count == P
    Multi-charger: add missing chargers until sum x == P
    """
    P = int(value(model.P))

    if not is_multi_charger_model(model):
        open_now = list_open_sites(model)
        missing = max(0, P - len(open_now))
        if missing <= 0:
            return model

        # deterministic: pick top missing by W1
        if greedy_mode == "deterministic":
            J = list(model.x.keys())
            I = range(len(demand_I))
            a = demand_I

            def W1(j):
                return sum(a[i] for i in I if (i, j) in model.Arcs)

            candidates = [j for j in J if j not in set(open_now)]
            candidates_sorted = sorted(candidates, key=W1, reverse=True)
            for j in candidates_sorted[:missing]:
                model.x[j].value = 1
            return model

        # weighted modes
        weight_mode = "W1" if greedy_mode == "weighted_W1" else "W2"
        return greedy_add_missing_units_binary(model, distIJ, demand_I, method_name, weight_mode, missing)

    # -------- multi-charger reconstruction: add chargers until sum x == P --------
    cur_total = total_chargers(model)
    missing = max(0, P - cur_total)
    if missing <= 0:
        sync_open_indicator(model)
        return model

    # weights based on W1 or W2-like score
    I = range(len(demand_I))
    a = demand_I
    J = list(model.J)

    def W1(j):
        return sum(a[i] for i in I if (i, j) in model.Arcs)

    # Use incremental adds (so W2 can penalize already-heavy sites)
    for _ in range(missing):
        if greedy_mode in ("deterministic", "weighted_W1"):
            weights = np.array([max(1e-12, W1(j)) for j in J], dtype=float)
        elif greedy_mode == "weighted_W2":
            # penalize already high x (anti-cluster by count)
            weights = np.array([max(1e-12, W1(j)) / (1.0 + charger_count(model, j)) for j in J], dtype=float)
        else:
            weights = np.array([max(1e-12, W1(j)) for j in J], dtype=float)

        probs = weights / weights.sum()
        pick = int(np.random.choice(J, p=probs))
        model.x[pick].value = charger_count(model, pick) + 1

    sync_open_indicator(model)
    return model


def greedy_add_missing_units_binary(model, distIJ, demand_I, method_name, weight_mode, k_add):
    """
    Binary helper: add k_add stations (x=1) using W1/W2.
    """
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
        else:  # W2
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
    if is_multi_charger_model(model):
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
