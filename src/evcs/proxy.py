import numpy as np

from evcs.utils import _u_to_capacity_array


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
