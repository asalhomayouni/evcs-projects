import numpy as np


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
