import numpy as np

from evcs.utils import _u_to_capacity_array


def compute_overflow_t0(U_dict, demand_t0, sorted_J_i, mu_ue, T, N):
    """
    Congestion overflow at period 0 under the UE capacity model.

    Each demand node is attracted to its nearest open in-range station
    (no capacity enforcement). The overflow is then:
        overflow = Σ_j max(0, attracted_demand_j - s_j_cumulative * mu_ue)

    This measures how much demand the greedy binary proxy "counts" but UE
    would not serve (it sends the overflow to no-option or a farther station).
    """
    s = np.zeros(N, dtype=float)
    for j in range(N):
        for t in range(T):
            s[j] += float(U_dict.get((j, t), 0))

    cap_j = s * float(mu_ue)
    attracted = np.zeros(N, dtype=float)

    for i in range(N):
        d_i = float(demand_t0[i])
        if d_i <= 1e-12:
            continue
        for j in sorted_J_i[i]:       # pre-sorted ascending dist, filtered to max_range
            if s[j] > 0:
                attracted[j] += d_i
                break                  # nearest-only: all of d_i goes to closest open station

    return float(np.sum(np.maximum(0.0, attracted - cap_j)))


def evaluate_u_congestion_proxy(
    U_dict,
    demand_TM,
    J_i_list,
    distIJ,
    Q_cap,
    T,
    N,
    cumulative_install,
    sorted_J_i,
    mu_ue,
    lambda_cong,
):
    """
    Congestion-aware proxy:
        binary_proxy(U) - lambda_cong * overflow_t0(U)

    binary_proxy uses T=6/Q_cap=20 (captures placement signal).
    overflow_t0 uses period-0 demand with UE capacity (s_j * mu_ue),
    penalising solutions whose stations would be over-subscribed in the UE.
    """
    bin_score = evaluate_u_numpy_greedy_binary(
        U_dict, demand_TM=demand_TM, J_i_list=J_i_list, distIJ=distIJ,
        Q_cap=Q_cap, T=T, N=N, cumulative_install=cumulative_install,
    )
    overflow = compute_overflow_t0(U_dict, demand_TM[0], sorted_J_i, mu_ue, T, N)
    return float(bin_score - lambda_cong * overflow)


def evaluate_u_ue_greedy(
    U_dict,
    demand_TM,
    sorted_J_i,
    station_cap,
    noopt_max_range,
    distIJ,
    T,
    N,
    cumulative_install=True,
):
    """
    UE-consistent greedy proxy. Port of Julia assign_demand_with_capacity.

    Fractional assignment: demand splits across nearest open stations.
    A demand unit goes to no-option if its nearest station is beyond
    noopt_max_range OR all in-range stations are full.

    sorted_J_i[i]: list of ALL j indices sorted by distIJ[i,j] ascending
                   (precomputed once, filtered to dist <= noopt_max_range).
    station_cap:   capacity per server (station_cap_per_server in Julia).
    """
    # Extract server counts (same logic as UEEvaluator._extract_servers)
    s = np.zeros(N, dtype=int)
    for j in range(N):
        if T <= 1:
            val = U_dict.get((j, 0), U_dict.get(j, 0))
        elif cumulative_install:
            val = sum(int(U_dict.get((j, t), 0)) for t in range(T))
        else:
            val = int(U_dict.get((j, T - 1), 0))
        s[j] = int(val)

    cap_rem = s.astype(float) * station_cap   # capacity vector

    covered = 0.0
    d_arr = demand_TM[0]                       # UE is single-period

    for i in range(N):
        d_i = float(d_arr[i])
        if d_i <= 1e-12:
            continue

        remaining = d_i
        for j in sorted_J_i[i]:               # pre-sorted, pre-filtered to <= noopt_max_range
            if s[j] == 0:
                continue
            avail = cap_rem[j]
            if avail <= 1e-12:
                continue
            flow = min(remaining, avail)
            cap_rem[j] -= flow
            remaining -= flow
            covered += flow
            if remaining <= 1e-12:
                break

    return float(covered)


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
