from __future__ import annotations
import random
from typing import Optional
import numpy as np


# --- helpers ---
def _dij(distIJ, i: int, j: int) -> float:
    try:
        return float(distIJ[i, j])
    except (TypeError, KeyError, IndexError):
        return float(distIJ[i][j])


def _sync_x_from_u(U_dict, N, T, cumulative_install, site_ub=None):
    x = {}
    for j in range(N):
        running = 0
        for t in range(T):
            ut = int(U_dict.get((j, t), 0))
            if cumulative_install:
                running += ut
                xv = running
            else:
                xv = ut
            if site_ub is not None:
                xv = min(xv, site_ub)
            x[j, t] = max(0, xv)
    return x


def _z_from_x(x_jt, N, T):
    return {(j, t): (1 if x_jt.get((j, t), 0) > 0 else 0) for j in range(N) for t in range(T)}


def _build_ji(arcs):
    Ji = {}
    for (i, j) in arcs:
        Ji.setdefault(int(i), []).append(int(j))
    return Ji


# --- greedy assign ---
def reassign_y_greedy_multi_pure(x_jt, z_jt, Q, demand_TM, arcs, M, N, T, distIJ, method_name):
    Ji_int = _build_ji(arcs)
    name = str(method_name).lower().strip()
    y = {}

    for t in range(T):
        a = {i: float(demand_TM[t, i]) for i in range(M)}
        cap_rem = {j: Q * float(x_jt.get((j, t), 0)) for j in range(N)}
        open_sites = {j for j in range(N) if z_jt.get((j, t), 0) > 0}

        for ii in sorted(range(M), key=lambda i: a[i], reverse=True):
            if a[ii] <= 1e-12:
                continue
            reachable = Ji_int.get(ii, [])
            open_reach = [j for j in reachable if j in open_sites]
            feasible = [j for j in open_reach if cap_rem[j] >= a[ii] - 1e-9]
            if not feasible:
                continue

            if name == "closest_only":
                chosen = min(feasible, key=lambda j: _dij(distIJ, ii, j))
            elif name == "closest_priority":
                chosen = min(feasible, key=lambda j: (_dij(distIJ, ii, j), cap_rem[j] - a[ii]))
            elif name == "system_optimum":
                chosen = min(feasible, key=lambda j: (a[ii] * _dij(distIJ, ii, j), cap_rem[j] - a[ii]))
            elif name == "uniform":
                chosen = random.choice(feasible)
            else:
                chosen = min(feasible, key=lambda j: (cap_rem[j] - a[ii], _dij(distIJ, ii, j)))

            y[ii, chosen, t] = 1
            cap_rem[chosen] -= a[ii]

    return y


# --- coverage ---
def covered_by_period_pure(y_ijt, demand_TM, arcs, M, T):
    Ji_int = _build_ji(arcs)
    cov = np.zeros(T, dtype=float)
    for t in range(T):
        for i in range(M):
            for j in Ji_int.get(i, []):
                if y_ijt.get((i, j, t), 0) >= 0.5:
                    cov[t] += float(demand_TM[t, i])
                    break
    return cov


# --- full eval ---
def full_eval_from_U_grb(U_dict, Q, inst, distIJ, policy, demand_TM=None, cumulative_install=True):
    arcs = list(inst["in_range"])
    M = len(inst["coords_I"])
    N = len(inst["coords_J"])

    raw = inst["demand_IT"] if demand_TM is None else demand_TM
    demand = np.asarray(raw, dtype=float)
    if demand.ndim != 2:
        raise ValueError(f"demand must be 2-D, got shape {demand.shape}")
    if demand.shape[1] == M:
        demand_arr = demand
    elif demand.shape[0] == M:
        demand_arr = demand.T
    else:
        raise ValueError(f"demand shape {demand.shape} incompatible with M={M}")
    T = demand_arr.shape[0]

    x_jt = _sync_x_from_u(U_dict, N, T, cumulative_install)
    z_jt = _z_from_x(x_jt, N, T)
    y_ijt = reassign_y_greedy_multi_pure(x_jt, z_jt, Q, demand_arr, arcs, M, N, T, distIJ, policy)
    cov = covered_by_period_pure(y_ijt, demand_arr, arcs, M, T)

    return float(np.sum(cov)), y_ijt, cov
