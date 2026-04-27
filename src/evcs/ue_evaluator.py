from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB
    from evcs.model_grb import _get_env
    _GRB_AVAILABLE = True
except ImportError:
    _GRB_AVAILABLE = False


# queue

def erlang_c(s: int, lam: float, mu: float) -> float:
    rho = lam / (s * mu)
    if rho >= 1.0:
        return 1.0
    a = lam / mu
    sum_terms = sum(a**k / math.factorial(k) for k in range(s))
    last_term = (a**s) / (math.factorial(s) * (1.0 - rho))
    return last_term / (sum_terms + last_term)


def waiting_time_mms(s: int, mu: float, lam: float) -> float:
    cap = s * mu
    if lam >= cap:
        return 1.0e12
    return erlang_c(s, lam, mu) / (cap - lam)


# pwl

def build_pwl_tables(
    N: int,
    s_max: int,
    mu: np.ndarray,
    cap: np.ndarray,
    lambda_max_global: float,
    N_bp: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lambda_hat = np.zeros((N, s_max, N_bp))
    w_arr      = np.zeros((N, s_max, N_bp))
    F_arr      = np.zeros((N, s_max, N_bp))

    for j in range(N):
        for si in range(s_max):
            s = si + 1
            local_max = min(lambda_max_global, float(cap[j, si]))
            grid = np.linspace(0.0, local_max, N_bp)
            lambda_hat[j, si, :] = grid
            for k in range(N_bp):
                w_arr[j, si, k] = waiting_time_mms(s, float(mu[j]), float(grid[k]))
            F_arr[j, si, 0] = 0.0
            for k in range(1, N_bp):
                h = grid[k] - grid[k - 1]
                F_arr[j, si, k] = (
                    F_arr[j, si, k - 1]
                    + 0.5 * (w_arr[j, si, k - 1] + w_arr[j, si, k]) * h
                )

    return lambda_hat, w_arr, F_arr


# evaluator

class UEEvaluator:

    def __init__(
        self,
        N: int,
        d: np.ndarray,
        tau: np.ndarray,
        mu,
        s_max: int,
        noopt_cost,
        alpha_wait: float = 1.0,
        N_bp: int = 100,
        lambda_scale: float = 1.0,
        eps: float = 1e-4,
        max_range: float = None,
        penalty_distance: float = 1.0,
    ):
        if not _GRB_AVAILABLE:
            raise ImportError("gurobipy is required for UEEvaluator")

        self.N = int(N)
        self.d = np.asarray(d, dtype=float)
        self.tau = np.asarray(tau, dtype=float)
        self.s_max = int(s_max)
        self.alpha_wait = float(alpha_wait)
        self.N_bp = int(N_bp)
        self.eps = float(eps)
        self.max_range = max_range              # hard distance cutoff on assignments (km)
        self.penalty_distance = float(penalty_distance)  # multiplier on tau in objective

        self._mu = np.full(N, float(mu)) if np.isscalar(mu) else np.asarray(mu, dtype=float)
        self._noopt = np.full(N, float(noopt_cost)) if np.isscalar(noopt_cost) else np.asarray(noopt_cost, dtype=float)

        si_vals = np.arange(1, s_max + 1, dtype=float)
        self._cap = np.maximum(np.outer(self._mu, si_vals) - eps, 1e-9)

        lambda_max_global = max(0.0, lambda_scale * float(np.sum(self.d)) - eps)
        self._lambda_hat, self._w, self._F = build_pwl_tables(
            self.N, self.s_max, self._mu, self._cap, lambda_max_global, self.N_bp,
        )

    def _extract_servers(self, U_dict: Dict, T: int, cumulative_install: bool) -> np.ndarray:
        s = np.zeros(self.N, dtype=int)
        for j in range(self.N):
            if T <= 1:
                val = U_dict.get((j, 0), U_dict.get(j, 0))
                s[j] = int(val)
            elif cumulative_install:
                s[j] = sum(int(U_dict.get((j, t), 0)) for t in range(T))
            else:
                s[j] = int(U_dict.get((j, T - 1), 0))
            s[j] = min(s[j], self.s_max)
        return s

    def _build_reach(self, open_j):
        """Return (reach_i, reach_j) dicts filtered by max_range."""
        N = self.N
        reach_i = {i: [] for i in range(N)}
        reach_j = {j: [] for j in open_j}
        for i in range(N):
            for j in open_j:
                if self.max_range is None or float(self.tau[i, j]) <= self.max_range:
                    reach_i[i].append(j)
                    reach_j[j].append(i)
        return reach_i, reach_j

    def evaluate(
        self,
        U_dict: Dict,
        T: int = 1,
        cumulative_install: bool = True,
    ) -> Tuple[float, None]:
        s_j = self._extract_servers(U_dict, T, cumulative_install)
        open_j = [j for j in range(self.N) if s_j[j] > 0]

        if not open_j:
            return 0.0, None

        N = self.N
        I = range(N)

        reach_i, reach_j = self._build_reach(open_j)
        valid_ij = [(i, j) for i in I for j in reach_i[i]]

        gm = gp.Model(env=_get_env())
        gm.setParam("OutputFlag", 0)

        y       = gm.addVars(valid_ij, lb=0.0, name="y") if valid_ij else {}
        y_noopt = gm.addVars(N, lb=0.0, name="yn")
        lam     = gm.addVars(open_j, lb=0.0, name="lam")
        phi     = gm.addVars(open_j, lb=0.0, name="phi")

        gm.update()

        # objective: distance (scaled) + no-option penalty + waiting cost
        gm.setObjective(
            gp.quicksum(
                self.penalty_distance * float(self.tau[i, j]) * y[i, j]
                for (i, j) in valid_ij
            )
            + gp.quicksum(self._noopt[i] * y_noopt[i] for i in I)
            + self.alpha_wait * gp.quicksum(phi[j] for j in open_j),
            GRB.MINIMIZE,
        )

        # C4: demand balance (only in-range stations available)
        for i in I:
            gm.addConstr(
                gp.quicksum(y[i, j] for j in reach_i[i]) + y_noopt[i] == float(self.d[i])
            )

        # C6: station total flow
        for j in open_j:
            gm.addConstr(lam[j] == gp.quicksum(y[i, j] for i in reach_j[j]))

        # C7: capacity ceiling
        for j in open_j:
            gm.addConstr(lam[j] <= float(self._cap[j, s_j[j] - 1]))

        # C8: piecewise-linear waiting cost
        for j in open_j:
            si = s_j[j] - 1
            for k in range(self.N_bp):
                gm.addConstr(
                    phi[j] >= float(self._F[j, si, k])
                    + float(self._w[j, si, k]) * (lam[j] - float(self._lambda_hat[j, si, k]))
                )

        gm.update()
        gm.optimize()

        total_lambda = 0.0
        if gm.SolCount > 0:
            total_lambda = sum(lam[j].X for j in open_j)

        gm.dispose()
        return float(total_lambda), None

    def dispose(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.dispose()
