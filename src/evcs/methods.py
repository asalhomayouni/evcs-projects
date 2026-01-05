# src/evcs/methods.py
from pyomo.environ import ConstraintList, Objective, maximize, value
import random
import numpy as np

# =========================================================
# Evaluate solution
# =========================================================
def evaluate_solution(m, distIJ, demand_I, method_name="closest_only"):
    """
    Evaluate the model using its Pyomo objective (m.obj).
    This ensures exact and heuristic scores are directly comparable.
    """
    obj_val = float(value(m.obj))

    total_demand = float(sum(demand_I))
    cov_term = sum(float(demand_I[i]) * (m.y[i, j].value or 0.0)
                   for (i, j) in m.Arcs)
    cov_pct = 100 * cov_term / total_demand if total_demand > 0 else 0.0

    return dict(
        covered_demand=obj_val,
        covered_pct=cov_pct,
    )


# =========================================================
# Policy definitions
# =========================================================
def add_closest_only(m, farther_of):
    m.closest_only = ConstraintList()
    for (i, j), farther in farther_of.items():
        m.closest_only.add(sum(m.y[ii, jj] for (ii, jj) in farther) <= 1 - m.x[j])
    print("→ Added 'closest_only' constraints.")
    return m


def add_closest_priority(m, distIJ):
    if hasattr(m, "obj"):
        m.del_component("obj")
    expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    expr_tiebreak = sum((1.0 - distIJ[i][j]) * m.y[i, j] for (i, j) in m.Arcs)
    m.add_component("obj", Objective(expr=expr_cov + 1e-3 * expr_tiebreak, sense=maximize))
    print("→ Added 'closest_priority' objective with distance tie-break.")
    return m


def add_system_optimum(m, distIJ, lambda_dist=0.1):
    if hasattr(m, "obj"):
        m.del_component("obj")
    expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    expr_dist = sum(distIJ[i][j] * m.y[i, j] for (i, j) in m.Arcs)
    m.add_component("obj", Objective(expr=expr_cov - lambda_dist * expr_dist, sense=maximize))
    print(f"→ Added 'system_optimum' hybrid objective (λ={lambda_dist}).")
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
    print("→ Added 'uniform' equal-split constraints.")
    return m


# =========================================================
# Greedy initializer (capacity-aware)
# =========================================================
def build_initial_solution(m, distIJ, mode="greedy", policy=None):
    """
    Builds an initial feasible solution consistent with the chosen policy
    and respecting station capacities.
    """
    I, J = sorted(m.I), sorted(m.J)
    P = int(m.P.value if hasattr(m.P, "value") else int(m.P()))
    Q = float(m.Q.value)
    if P <= 0:
        return m

    # Reset
    for j in J:
        m.x[j].value = 0.0
    for (i, j) in m.Arcs:
        m.y[i, j].value = 0.0

    # --- Choose open stations
    J_shuffled = list(J)
    

    if policy == "system_optimum":
        score = {j: sum(distIJ[i][j] for i in I if (i, j) in m.Arcs) for j in J}
        open_list = sorted(score, key=score.get)[:P]
    elif policy == "closest_priority":
        score = {j: -sum(distIJ[i][j] for i in I if (i, j) in m.Arcs) for j in J}
        open_list = sorted(score, key=score.get, reverse=True)[:P]
    elif policy == "uniform":
        step = max(1, len(J_shuffled) // P)
        open_list = [J_shuffled[k] for k in range(0, len(J_shuffled), step)][:P]
    else:
        
        open_list = J[:P]


    for j in open_list:
        m.x[j].value = 1.0

    # --- Assign y[i,j] (respect capacity)
    a = {i: float(m.a[i]) for i in I}
    cap_rem = {j: Q * (m.x[j].value or 0.0) for j in J}

    for i in I:
        reachable = [j for j in J if (i, j) in m.Arcs]
        open_sites = [j for j in reachable if (m.x[j].value or 0.0) > 0.5]
        if not reachable:
            continue

        if policy == "uniform":
            frac = 1.0 / len(reachable)
            for j in reachable:
                m.y[i, j].value = frac * (m.x[j].value or 0.0)
        else:
            if not open_sites:
                continue
            sorted_sites = sorted(open_sites, key=lambda jj: distIJ[i][jj])
            for j in sorted_sites:
                if cap_rem[j] >= a[i] - 1e-9:
                    m.y[i, j].value = 1.0
                    cap_rem[j] -= a[i]
                    break

    print(f"✅ Greedy initialized with {len(open_list)} stations ({policy}).")
    return m

from pyomo.environ import value

def build_initial_solution_smart(model, distIJ, method_name="closest_only"):
    """
    Deterministic greedy initializer:
    - Start empty
    - Repeatedly open the station that produces the largest improvement
    - Until P stations are open
    """

    # Extract basic sets
    I = sorted({i for i, _ in model.Arcs})
    J = sorted({j for _, j in model.Arcs})

    # Capacity and demand parameters
    Q = float(model.Q.value)
    a = {i: float(model.a[i]) for i in I}

    # Find P in the model
    try:
        P = int(value(model.P))
    except:
        raise RuntimeError("Model has no parameter P defined.")

    # Helper: reassign_y (same as in local_search)
    def reassign_y(m):
        cap_rem = {j: Q * (m.x[j].value or 0.0) for j in J}

        # reset y
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
                sorted_sites = sorted(open_sites, key=lambda jj: distIJ[i][jj])
                for j in sorted_sites:
                    if cap_rem[j] >= a[i] - 1e-9:
                        m.y[i, j].value = 1.0
                        cap_rem[j] -= a[i]
                        break

    def objective(m):
        return float(value(m.obj))

    # Start with everything closed
    for j in J:
        model.x[j].value = 0.0
    for (i, j) in model.Arcs:
        model.y[i, j].value = 0.0

    reassign_y(model)
    current_score = objective(model)

    # Greedy: open P stations
    for k in range(P):

        best_gain = -1e18
        best_j = None

        closed_sites = [j for j in J if (model.x[j].value or 0.0) < 0.5]

        for j in closed_sites:
            model.x[j].value = 1.0
            reassign_y(model)
            new_score = objective(model)
            gain = new_score - current_score

            if gain > best_gain + 1e-9:
                best_gain = gain
                best_j = j

            # revert
            model.x[j].value = 0.0
            reassign_y(model)

        if best_j is None:
            break

        # Apply best station
        model.x[best_j].value = 1.0
        reassign_y(model)
        current_score += best_gain

    reassign_y(model)
    return model

# =========================================================
# Weighted Greedy Initializer (W1)
# =========================================================
# =========================================================
# Weighted Greedy Initializer (W1 / W2)
# =========================================================
def build_initial_solution_weighted(
    model, distIJ, demand_I,
    method_name="closest_only",
    weight_mode="W1",
):
    import numpy as np, random
    from pyomo.environ import value

    # Sets & params
    I = sorted({i for i, _ in model.Arcs})
    J = sorted({j for _, j in model.Arcs})
    Q = float(model.Q.value)
    a = {i: float(model.a[i]) for i in I}
    P = int(value(model.P))

    # Reset x,y
    for j in J:
        model.x[j].value = 0.0
    for (i, j) in model.Arcs:
        model.y[i, j].value = 0.0

    # ---- W1: total demand in range of j ----
    W1 = {}
    for j in J:
        W1[j] = sum(a[i] for i in I if (i, j) in model.Arcs)

    chosen = []

    if weight_mode == "W1":
        total = sum(W1.values())
        if total <= 1e-12:
            chosen = random.sample(J, k=min(P, len(J)))
        else:
            probs = [W1[j]/total for j in J]
            chosen = list(np.random.choice(J, size=P, replace=False, p=probs))

    elif weight_mode == "W2":
        # Anti-clustering: penalize candidates close to already chosen sites
        remaining = set(J)
        # radius you can tune; median pairwise distance is a decent default
        # if distIJ is dict-of-dicts, this fallback uses a rough scalar
        try:
            all_d = []
            for i in J:
                for j in J:
                    if i != j:
                        all_d.append(distIJ[i][j])
            D_cluster = float(np.median(all_d)) * 0.5 if all_d else 1.0
        except Exception:
            D_cluster = 1.0

        for _ in range(min(P, len(J))):
            W2 = {}
            for j in remaining:
                near_cnt = sum(1 for jc in chosen if distIJ[j][jc] <= D_cluster)
                W2[j] = W1[j] / (1 + near_cnt)

            total = sum(W2.values())
            if total <= 1e-12:
                pick = random.choice(list(remaining))
            else:
                pool = list(remaining)
                probs = [W2[j]/total for j in pool]
                pick = np.random.choice(pool, p=probs)

            chosen.append(pick)
            remaining.remove(pick)


    else:
        raise ValueError("weight_mode must be 'W1' or 'W2'")

    # Open chosen sites
    for j in chosen:
        model.x[j].value = 1.0

    # Capacity-aware assignment (same rule as LS)
    cap_rem = {j: Q * (model.x[j].value or 0.0) for j in J}
    for i in I:
        reach = [j for j in J if (i, j) in model.Arcs]
        open_sites = [j for j in reach if (model.x[j].value or 0.0) > 0.5]
        if not reach or not open_sites:
            continue
        sorted_sites = sorted(open_sites, key=lambda jj: distIJ[i][jj])
        for j in sorted_sites:
            if cap_rem[j] >= a[i] - 1e-9:
                model.y[i, j].value = 1.0
                cap_rem[j] -= a[i]
                break
    
     
    print(f"✅ Weighted greedy ({weight_mode}) initialized with {len(chosen)} stations.")
    return model

# ========================================================
# Adaptive Multi-Neighborhood Local Search
# =========================================================
def local_search(model, distIJ, in_range, Ji, Ij, farther_of,
                 method_name="closest_only", max_iter=100,
                 improvement_rule="first", try_order="seq",
                 logger=None):

    import random
    from pyomo.environ import value

    # --- Basic sets and parameters ---
    I = sorted({i for i, _ in model.Arcs})
    J = sorted({j for _, j in model.Arcs})
    Q = float(model.Q.value)
    a = {i: float(model.a[i]) for i in I}

    # -------------------------
    # y Reassignment function
    # -------------------------
    def reassign_y(m):
        cap_rem = {j: Q * (m.x[j].value or 0.0) for j in J}

        # reset all y
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
                sorted_sites = sorted(open_sites, key=lambda jj: distIJ[i][jj])
                for j in sorted_sites:
                    if cap_rem[j] >= a[i] - 1e-9:
                        m.y[i, j].value = 1.0
                        cap_rem[j] -= a[i]
                        break

    # -------------------------
    def objective(m):
        return float(value(m.obj))

    # -------------------------
    # Log initial greedy BEFORE reassignment
    # -------------------------


    reassign_y(model)
    best_score = objective(model)

    print(f"\n🔵 Local Search start ({method_name}), initial obj={best_score:.3f}")
     

     
    if logger:
        logger.log_iter(0, move_type="init",
                        obj_value=best_score,
                        detail="initial",
                        accepted=1)

    # -------------------------
    # Neighborhood sequence
    # -------------------------
    neighborhoods = ["open_close", "merge", "shift"]
    current_move = 0
    max_no_improve = 5
    no_improve_count = 0

    # -------------------------
    # Order selector
    # -------------------------
    def get_order(seq):
        seq = list(seq)
        if try_order == "random":
            random.shuffle(seq)
        return seq

    # =========================================================
    # BEST improvement helper
    # =========================================================
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

            if logger:
                logger.log_iter(it+1, move_type=mv_name,
                                obj_value=best_score,
                                detail=str(best_mv),
                                accepted=1)

            # print(f"Iter {it+1:02d}: [BEST] {mv_name} {best_mv} → {best_score:.3f}")
            return True
        return False

    # =========================================================
    # Main LS loop
    # =========================================================
    for it in range(max_iter):

        improved = False
        move_type = neighborhoods[current_move]

        open_sites = [j for j in J if (model.x[j].value or 0.0) > 0.5]
        closed_sites = [j for j in J if j not in open_sites]

        # =====================================================
        # 1) OPEN-CLOSE neighborhood
        # =====================================================
        if move_type == "open_close":

            # Collect all candidates for BEST improvement
            if improvement_rule == "best":
                candidates = [(j_close, j_open)
                              for j_close in get_order(open_sites)
                              for j_open in get_order(closed_sites)]

                def apply(data):
                    jc, jo = data
                    model.x[jc].value = 0
                    model.x[jo].value = 1

                def revert(data):
                    jc, jo = data
                    model.x[jc].value = 1
                    model.x[jo].value = 0

                improved = apply_best_improvement(candidates, apply, revert,
                                                  it, "open_close")

            # FIRST improvement (your existing behavior)
            else:
                for j_close in get_order(open_sites):
                    for j_open in get_order(closed_sites):
                        model.x[j_close].value = 0
                        model.x[j_open].value = 1
                        reassign_y(model)
                        new_score = objective(model)

                        if new_score > best_score + 1e-6:
                            best_score = new_score

                            if logger:
                                logger.log_iter(it+1, "open_close",
                                                best_score,
                                                f"{j_close}->{j_open}",
                                                accepted=1)

                            # print(f"Iter {it+1:02d}: open/close {j_close}->{j_open} → {best_score:.3f}")
                            improved = True
                            break

                        # revert
                        model.x[j_close].value = 1
                        model.x[j_open].value = 0
                        reassign_y(model)
                    if improved:
                        break

        # =====================================================
        # 2) MERGE neighborhood
        # =====================================================
        elif move_type == "merge":
            near_pairs = [(j1, j2) for j1 in open_sites
                                   for j2 in open_sites
                                   if j1 != j2 and distIJ[j1][j2] < 4.0]

            if improvement_rule == "best":

                candidates = get_order(near_pairs)

                def apply(data):
                    j1, j2 = data
                    model.x[j1].value = 0

                def revert(data):
                    j1, j2 = data
                    model.x[j1].value = 1

                improved = apply_best_improvement(candidates, apply, revert,
                                                  it, "merge")

            else:
                for j1, j2 in get_order(near_pairs):
                    model.x[j1].value = 0
                    reassign_y(model)
                    new_score = objective(model)

                    if new_score > best_score + 1e-6:
                        best_score = new_score
                        if logger:
                            logger.log_iter(it+1, "merge", best_score, f"{j1}->{j2}", accepted=1)
                        # print(f"Iter {it+1:02d}: merge {j1}->{j2} → {best_score:.3f}")
                        improved = True
                        break

                    model.x[j1].value = 1
                    reassign_y(model)

        # =====================================================
        # 3) SHIFT neighborhood
        # =====================================================
        elif move_type == "shift":
            shift_pairs = [(j1, j2) for j1 in open_sites
                                     for j2 in closed_sites
                                     if distIJ[j1][j2] < 6.0]

            if improvement_rule == "best":

                candidates = get_order(shift_pairs)

                def apply(data):
                    j1, j2 = data
                    model.x[j1].value = 0
                    model.x[j2].value = 1

                def revert(data):
                    j1, j2 = data
                    model.x[j1].value = 1
                    model.x[j2].value = 0

                improved = apply_best_improvement(candidates, apply, revert,
                                                  it, "shift")

            else:
                for j1, j2 in get_order(shift_pairs):
                    model.x[j1].value = 0
                    model.x[j2].value = 1
                    reassign_y(model)
                    new_score = objective(model)

                    if new_score > best_score + 1e-6:
                        best_score = new_score

                        if logger:
                            logger.log_iter(it+1, "shift",
                                            best_score, f"{j1}->{j2}", accepted=1)

                        # print(f"Iter {it+1:02d}: shift {j1}->{j2} → {best_score:.3f}")
                        improved = True
                        break

                    model.x[j1].value = 1
                    model.x[j2].value = 0
                    reassign_y(model)

        # =====================================================
        # Neighborhood switching
        # =====================================================
        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                no_improve_count = 0
                current_move += 1
                if current_move >= len(neighborhoods):
                    # print(f"🔚 No improvement in any neighborhood. Stopping at iter {it+1}.")
                    break
                # print(f"↪️ Switching to next neighborhood: {neighborhoods[current_move].upper()}")

    # =====================================================
    # print(f"✅ Final objective ({method_name}) = {best_score:.3f}")
    if logger:
        logger.finish_run(best_score)

    return model

# =========================================================
# Simulated Annealing Heuristic
def simulated_annealing(model, distIJ, in_range, Ji, Ij,
                        method_name="closest_only",
                        max_iter=200, T0=1.0, alpha=0.95,
                        Tmin=1e-3, L=20,
                        logger=None):
    import math, random
    from pyomo.environ import value

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
                sorted_sites = sorted(open_sites, key=lambda jj: distIJ[i][jj])
                for j in sorted_sites:
                    if cap_rem[j] >= a[i] - 1e-9:
                        m.y[i, j].value = 1.0
                        cap_rem[j] -= a[i]
                        break

    def objective(m):
        return float(value(m.obj))

    # Initialization
    reassign_y(model)
    current_model = model
    current_score = objective(current_model)
    best_model = current_model.clone()
    best_score = current_score

    if logger:
        logger.log_iter(0, move_type="init", obj_value=best_score, temp=T0, accepted=1)

    print(f"\n🔥 Simulated Annealing start ({method_name}) | initial obj={best_score:.3f}")

    T = T0
    iteration = 0

    while T > Tmin and iteration < max_iter:
        for _ in range(L):
            iteration += 1
            open_sites = [j for j in J if (current_model.x[j].value or 0.0) > 0.5]
            closed_sites = [j for j in J if j not in open_sites]
            if not open_sites or not closed_sites:
                continue
            j_close = random.choice(open_sites)
            j_open = random.choice(closed_sites)
            current_model.x[j_close].value = 0.0
            current_model.x[j_open].value = 1.0
            reassign_y(current_model)
            new_score = objective(current_model)
            delta = new_score - current_score
            accept = delta >= 0 or random.random() < math.exp(delta / max(T, 1e-9))
            if accept:
                current_score = new_score
                if new_score > best_score + 1e-9:
                    best_score = new_score
                    best_model = current_model.clone()
                    print(f"Iter {iteration:03d}: SA improved → {best_score:.3f} (T={T:.4f})")
                if logger:
                    logger.log_iter(iteration, move_type="swap", obj_value=current_score, temp=T, accepted=1, detail=f"{j_close}->{j_open}")
            else:
                current_model.x[j_close].value = 1.0
                current_model.x[j_open].value = 0.0
                reassign_y(current_model)
                if logger:
                    logger.log_iter(iteration, move_type="swap", obj_value=current_score, temp=T, accepted=0, detail=f"revert {j_close}->{j_open}")
            if iteration >= max_iter:
                break
        T *= alpha

    print(f"✅ Final SA objective ({method_name}) = {best_score:.3f}")
    if logger:
        logger.finish_run(best_score)
    return best_model

# =========================================================
# Helper
# =========================================================
def compute_farther(distIJ, in_range, Ji):
    farther_of = {}
    if distIJ is None:
        return farther_of
    for (i, j) in in_range:
        dij = distIJ[i][j]
        jlist = Ji.get(i, [])
        farther = [(i, k) for k in jlist if distIJ[i][k] > dij + 1e-6]
        if farther:
            farther_of[(i, j)] = farther
    return farther_of


# =========================================================
# Policy Wrapper
# =========================================================
def apply_method(m, method_name, distIJ, in_range, Ji, Ij, farther_of, verbose=False):
    """Attach the correct policy logic and objective to the model."""
    name = str(method_name).lower()

    if hasattr(m, "obj"):
        try:
            m.del_component("obj")
        except Exception:
            pass

    if name == "closest_only":
        add_closest_only(m, farther_of)
        expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
        m.add_component("obj", Objective(expr=expr_cov, sense=maximize))
        if verbose:
            print("→ Closest-only policy applied.")

    elif name == "closest_priority":
        expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
        expr_tiebreak = sum((1.0 - distIJ[i][j]) * m.y[i, j] for (i, j) in m.Arcs)
        m.add_component("obj", Objective(expr=expr_cov + 1e-3 * expr_tiebreak, sense=maximize))
        if verbose:
            print("→ Closest-priority policy applied.")

    elif name == "system_optimum":
        expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
        expr_dist = sum(distIJ[i][j] * m.y[i, j] for (i, j) in m.Arcs)
        m.add_component("obj", Objective(expr=expr_cov - 0.1 * expr_dist, sense=maximize))
        if verbose:
            print("→ System-optimum policy applied.")

    elif name == "uniform":
        add_uniform_allocation_constraints(m)
        if verbose:
            print("→ Uniform equal split constraints applied.")

    else:
        expr_cov = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
        m.add_component("obj", Objective(expr=expr_cov, sense=maximize))
        if verbose:
            print(f"⚠️ Default coverage objective (policy={method_name}).")

    return m

def destroy_partial(model, k):
    """
    Remove k open stations (biased to remove weakest ones).
    Leaves x[j]==0 and y[i,j]==0 for removed j.
    """
    import numpy as np

    Jopen = list_open_sites(model)
    if not Jopen:
        return model

    k = min(k, len(Jopen))

    # You can choose random removal:
    to_remove = np.random.choice(Jopen, size=k, replace=False)

    # Remove selected sites
    for j in to_remove:
        model.x[j].value = 0.0
        for (i, jj) in model.y:
            if jj == j:
                model.y[i, jj].value = 0.0

    return model


# ================================================================
# ===  DESTRUCTION + RECONSTRUCTION (D&R) SUPPORT FUNCTIONS     ===
# ================================================================

def list_open_sites(model):
    """Return list of j where x[j] = 1."""
    return [j for j in model.x if (model.x[j].value or 0) > 0.5]


def destroy_partial(model, k):
    """
    Remove k open stations by setting x[j]=0 and clearing y[i,j].
    This does NOT rebuild anything; reconstruction happens later.
    """
    import numpy as np

    Jopen = list_open_sites(model)
    if not Jopen:
        return model

    k = min(k, len(Jopen))
    to_remove = np.random.choice(Jopen, size=k, replace=False)

    for j in to_remove:
        # close the station
        model.x[j].value = 0.0
        # clear its assignments
        for (i, jj) in model.y:
            if jj == j:
                model.y[i, jj].value = 0.0

    return model


def greedy_add_missing_sites(model, distIJ, demand_I, D, method_name, weight_mode, k_add):
    """
    Append-only greedy: ADD k_add new stations WITHOUT clearing existing ones.
    """
    import numpy as np

    J = list(model.x.keys())
    open_now = set(list_open_sites(model))
    remaining = [j for j in J if j not in open_now]

    if hasattr(demand_I, "keys"):
        I = list(demand_I.keys())
    else:
        I = range(len(demand_I))
    a = demand_I

    # Base W1
    def W1(j):
        total = 0.0
        for i in I:
            if (i, j) in model.Arcs:
                total += a[i]
        return max(1e-12, total)

    for _ in range(min(k_add, len(remaining))):
        weights = {}

        if weight_mode == "W1":
            for j in remaining:
                weights[j] = W1(j)

        elif weight_mode == "W2":
            for j in remaining:
                near = sum(1 for c in open_now if distIJ[j][c] <= D)
                weights[j] = W1(j) / (1.0 + near)

        # sample
        total = sum(weights.values())
        if total <= 1e-12:
            pick = np.random.choice(remaining)
        else:
            probs = [weights[j]/total for j in remaining]
            pick = np.random.choice(remaining, p=probs)

        # open it
        model.x[pick].value = 1.0
        open_now.add(pick)
        remaining.remove(pick)

    return model



def reconstruction_greedy(model, distIJ, demand_I, D, method_name, greedy_mode):
    """
    Rebuild missing stations until #open = P, using W1 or W2.
    """
    P = int(model.P.value)
    open_now = list_open_sites(model)
    missing = P - len(open_now)

    if missing <= 0:
        return model

    if greedy_mode == "weighted_W2":
        return greedy_add_missing_sites(model, distIJ, demand_I, D, method_name, "W2", missing)

    elif greedy_mode == "weighted_W1":
        return greedy_add_missing_sites(model, distIJ, demand_I, D, method_name, "W1", missing)

    elif greedy_mode == "deterministic":
        return greedy_add_missing_sites(model, distIJ, demand_I, D, method_name, "W1", missing)

    else:
        raise ValueError(f"Unknown greedy_mode={greedy_mode}")




