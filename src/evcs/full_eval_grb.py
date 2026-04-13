# src/evcs/full_eval_grb.py
"""
Pure numpy/dict evaluation pipeline — no Pyomo or GRB model objects.

This mirrors full_eval.py but operates entirely on plain Python dicts and
numpy arrays, making it independent of both Pyomo and gurobipy at runtime.

Public API
----------
reassign_y_greedy_multi_pure(x_jt, z_jt, Q, demand_TM, arcs, M, N, T,
                              distIJ, method_name)
    → y_ijt dict: (i,j,t) → 0/1

covered_by_period_pure(y_ijt, demand_TM, arcs, M, T)
    → np.ndarray shape (T,)

full_eval_from_U_grb(U_dict, Q, inst, distIJ, policy,
                     demand_TM=None, cumulative_install=True)
    → (total_score: float, y_ijt: dict, cov_per_period: np.ndarray)
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np


# =========================================================
# Internal helpers
# =========================================================
def _dij(distIJ, i: int, j: int) -> float:
    """Robust distance getter: supports ndarray, nested list, or dict."""
    try:
        return float(distIJ[i, j])
    except (TypeError, KeyError, IndexError):
        return float(distIJ[i][j])


def _sync_x_from_u(
    U_dict: dict,
    N: int,
    T: int,
    cumulative_install: bool,
    site_ub: Optional[int] = None,
) -> dict:
    """
    Compute x_jt[(j,t)] = cumulative (or per-period) charger count from U_dict.
    Mirrors sync_x_from_u_multi in methods.py.
    """
    x: dict = {}
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


def _z_from_x(x_jt: dict, N: int, T: int) -> dict:
    """z_jt[(j,t)] = 1 if x_jt[(j,t)] > 0 else 0."""
    return {
        (j, t): (1 if x_jt.get((j, t), 0) > 0 else 0)
        for j in range(N)
        for t in range(T)
    }


def _build_ji(arcs) -> dict:
    """Build Ji_int: i → [j, …] from arc list."""
    Ji: dict = {}
    for (i, j) in arcs:
        Ji.setdefault(int(i), []).append(int(j))
    return Ji


# =========================================================
# Greedy assignment (pure dict version)
# =========================================================
def reassign_y_greedy_multi_pure(
    x_jt: dict,
    z_jt: dict,
    Q: float,
    demand_TM: np.ndarray,
    arcs: list,
    M: int,
    N: int,
    T: int,
    distIJ,
    method_name: str,
) -> dict:
    """
    Pure-Python greedy y assignment.  Mirrors methods.reassign_y_greedy_multi
    but works on plain dicts instead of Pyomo model objects.

    Parameters
    ----------
    x_jt, z_jt   : dicts (j,t) → int/float  (charger counts and open flags)
    Q             : demand capacity per charger
    demand_TM     : numpy array shape (T, M)
    arcs          : list of (i, j) tuples
    M, N, T       : instance sizes
    distIJ        : distance matrix (ndarray or nested list)
    method_name   : "closest_only" | "closest_priority" | "system_optimum"
                    | "uniform" | anything else → best-fit

    Returns
    -------
    y_ijt : dict  (i, j, t) → 0 or 1
    """
    Ji_int = _build_ji(arcs)
    name = str(method_name).lower().strip()
    y: dict = {}

    for t in range(T):
        a = {i: float(demand_TM[t, i]) for i in range(M)}
        cap_rem = {j: Q * float(x_jt.get((j, t), 0)) for j in range(N)}
        open_sites = {j for j in range(N) if z_jt.get((j, t), 0) > 0}

        # highest demand first
        for ii in sorted(range(M), key=lambda i: a[i], reverse=True):
            if a[ii] <= 1e-12:
                continue
            reachable = Ji_int.get(ii, [])
            if not reachable:
                continue
            open_reach = [j for j in reachable if j in open_sites]
            if not open_reach:
                continue
            feasible = [j for j in open_reach if cap_rem[j] >= a[ii] - 1e-9]
            if not feasible:
                continue

            if name == "closest_only":
                chosen = min(feasible, key=lambda j: _dij(distIJ, ii, j))
            elif name == "closest_priority":
                chosen = min(
                    feasible,
                    key=lambda j: (_dij(distIJ, ii, j), cap_rem[j] - a[ii]),
                )
            elif name == "system_optimum":
                chosen = min(
                    feasible,
                    key=lambda j: (
                        a[ii] * _dij(distIJ, ii, j),
                        cap_rem[j] - a[ii],
                    ),
                )
            elif name == "uniform":
                chosen = random.choice(feasible)
            else:  # best-fit
                chosen = min(
                    feasible,
                    key=lambda j: (cap_rem[j] - a[ii], _dij(distIJ, ii, j)),
                )

            y[ii, chosen, t] = 1
            cap_rem[chosen] -= a[ii]

    return y


# =========================================================
# Coverage computation (pure dict version)
# =========================================================
def covered_by_period_pure(
    y_ijt: dict,
    demand_TM: np.ndarray,
    arcs: list,
    M: int,
    T: int,
) -> np.ndarray:
    """
    Compute covered demand per period from a y_ijt assignment dict.
    Mirrors full_eval.covered_by_period but takes a dict instead of a model.

    Returns
    -------
    cov : np.ndarray shape (T,)
    """
    Ji_int = _build_ji(arcs)
    cov = np.zeros(T, dtype=float)

    for t in range(T):
        for i in range(M):
            for j in Ji_int.get(i, []):
                if y_ijt.get((i, j, t), 0) >= 0.5:
                    cov[t] += float(demand_TM[t, i])
                    break  # at most one assignment per (i, t)

    return cov


# =========================================================
# Full evaluation pipeline
# =========================================================
def full_eval_from_U_grb(
    U_dict: dict,
    Q: float,
    inst: dict,
    distIJ,
    policy: str,
    demand_TM: Optional[np.ndarray] = None,
    cumulative_install: bool = True,
) -> tuple:
    """
    Full greedy evaluation of a charger-installation decision U_dict.
    Drop-in replacement for full_eval.full_eval_from_U that needs no Pyomo
    model template — pass Q explicitly instead.

    Parameters
    ----------
    U_dict           : dict (j, t) → int  chargers installed at site j in period t
    Q                : demand capacity per charger  (float)
    inst             : instance dict with keys:
                         demand_IT, coords_I, coords_J, in_range, Ji, Ij
    distIJ           : pairwise distance matrix
    policy           : assignment policy name string
    demand_TM        : optional (T, M) demand array override
                       (falls back to inst["demand_IT"])
    cumulative_install: whether chargers accumulate across periods

    Returns
    -------
    (total_score: float,  y_ijt: dict,  cov_per_period: np.ndarray)
    """
    arcs = list(inst["in_range"])
    M = len(inst["coords_I"])
    N = len(inst["coords_J"])

    # --- normalise demand to (T, M) ---
    raw = inst["demand_IT"] if demand_TM is None else demand_TM
    demand = np.asarray(raw, dtype=float)
    if demand.ndim != 2:
        raise ValueError(f"demand must be 2-D, got shape {demand.shape}")
    if demand.shape[1] == M:
        demand_arr = demand          # already (T, M)
    elif demand.shape[0] == M:
        demand_arr = demand.T        # (M, T) → (T, M)
    else:
        raise ValueError(
            f"demand shape {demand.shape} incompatible with M={M}"
        )
    T = demand_arr.shape[0]

    # --- compute x and z from U ---
    x_jt = _sync_x_from_u(U_dict, N, T, cumulative_install)
    z_jt = _z_from_x(x_jt, N, T)

    # --- greedy assignment ---
    y_ijt = reassign_y_greedy_multi_pure(
        x_jt, z_jt, Q, demand_arr, arcs, M, N, T, distIJ, policy
    )

    # --- coverage ---
    cov = covered_by_period_pure(y_ijt, demand_arr, arcs, M, T)
    score = float(np.sum(cov))

    return score, y_ijt, cov
