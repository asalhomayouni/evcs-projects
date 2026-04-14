from __future__ import annotations
import random
from typing import Optional
import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
    from evcs.model_grb import _get_env
    _GRB_AVAILABLE = True
except ImportError:
    _GRB_AVAILABLE = False


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


class GRBEvaluator:
    """Gurobi y-assignment model built once and reused across DR iterations.

    Given a fixed placement U_dict (which determines x[j,t]), solves for the
    optimal demand-coverage assignment y.  One license checkout for the whole
    DR run; dispose() is called once at the end.

    Usage::
        with GRBEvaluator(M, N, T, arcs, demand_TM, Q) as ev:
            score0, _ = ev.evaluate(U_init)
            for it in range(max_iter):
                score, cov = ev.evaluate(U_candidate)
        # dispose() called automatically on __exit__
    """

    def __init__(self, M, N, T, arcs, demand_TM, Q, cumulative_install=True):
        if not _GRB_AVAILABLE:
            raise ImportError("gurobipy is required for GRBEvaluator")

        self.M = M
        self.N = N
        self.T = T
        self.Q = float(Q)
        self.cumulative_install = cumulative_install
        self._arcs = list(arcs)

        d = np.asarray(demand_TM, dtype=float)  # (T, M)
        self._a = {(i, t): float(d[t, i]) for i in range(M) for t in range(T)}

        # precompute arc lookups used in evaluate()
        arcs_by_i = {i: [] for i in range(M)}
        arcs_by_j = {j: [] for j in range(N)}
        for (i, j) in self._arcs:
            arcs_by_i[i].append(j)
            arcs_by_j[j].append(i)
        self._arcs_by_i = arcs_by_i

        # --- build the model once (one license checkout) ---
        gm = gp.Model(env=_get_env())
        gm.setParam("OutputFlag", 0)

        arc_t_keys = [(i, j, t) for (i, j) in self._arcs for t in range(T)]
        # CONTINUOUS with ub=1 — the constraint matrix is totally unimodular
        # (bipartite assignment), so the LP relaxation is always integer-valued.
        # This makes each evaluate() call an LP instead of a MIP: ~50x faster.
        y = gm.addVars(arc_t_keys, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="y")
        gm.update()

        # objective: maximise covered demand
        gm.setObjective(
            gp.quicksum(self._a[i, t] * y[i, j, t] for (i, j, t) in arc_t_keys),
            GRB.MAXIMIZE,
        )

        # each demand node served by at most one site per period
        for i in range(M):
            js = arcs_by_i[i]
            if not js:
                continue
            for t in range(T):
                gm.addConstr(gp.quicksum(y[i, j, t] for j in js) <= 1)

        # capacity constraints — RHS updated via c.RHS each evaluate() call
        cap_constrs = {}
        for j in range(N):
            i_list = arcs_by_j[j]
            if not i_list:
                continue
            for t in range(T):
                c = gm.addConstr(
                    gp.quicksum(self._a[i, t] * y[i, j, t] for i in i_list) <= 0.0
                )
                cap_constrs[j, t] = c

        gm.update()

        self._gm = gm
        self._y = y
        self._cap_constrs = cap_constrs

    def evaluate(self, U_dict):
        """Fix placement U_dict, re-optimize y, return (total_score, cov_per_period).

        Only bound updates + RHS changes happen — the model structure is
        never rebuilt.  Gurobi warm-starts from the previous solution.
        """
        x_jt = _sync_x_from_u(U_dict, self.N, self.T, self.cumulative_install)

        # fix y upper bounds: 0 if site j is closed at t, 1 if open
        for (i, j) in self._arcs:
            for t in range(self.T):
                self._y[i, j, t].UB = 1.0 if x_jt.get((j, t), 0) > 0 else 0.0

        # update capacity RHS: Q * x[j,t]
        for (j, t), c in self._cap_constrs.items():
            c.RHS = self.Q * float(x_jt.get((j, t), 0))

        self._gm.update()
        self._gm.optimize()

        cov = np.zeros(self.T, dtype=float)
        if self._gm.SolCount > 0:
            for t in range(self.T):
                for i in range(self.M):
                    for j in self._arcs_by_i.get(i, []):
                        if self._y[i, j, t].X > 0.5:
                            cov[t] += self._a[i, t]
                            break

        return float(np.sum(cov)), cov

    def dispose(self):
        """Release the Gurobi model — one license return for the whole DR run."""
        self._gm.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.dispose()
