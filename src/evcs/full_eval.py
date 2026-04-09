import numpy as np

from evcs.utils import _apply_u_matrix
from evcs.methods import sync_solution_state, reassign_y_greedy_multi


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
