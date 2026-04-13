# src/evcs/model_grb.py
"""
gurobipy-native model builders that expose a Pyomo-compatible interface.

GRBModelWrapper wraps a gurobipy.Model and provides the same attribute-access
pattern (.I, .J, .T, .Arcs, .x[j,t].value, .y[i,j,t].value, …) as a Pyomo
ConcreteModel, so methods.py / full_eval.py / dr.py work without modification.

Public API
----------
build_base_model_grb(...)          mirrors model.build_base_model
build_multi_period_model_grb(...)  mirrors model.build_multi_period_model
GRBModelWrapper                    the returned wrapper type
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import gurobipy as gp
from gurobipy import GRB


# =========================================================
# Internal key normaliser
# =========================================================
def _nk(k) -> Any:
    """Normalise dict key to use plain ints (avoids numpy-int mismatches)."""
    if isinstance(k, tuple):
        return tuple(int(x) for x in k)
    return int(k)


# =========================================================
# Proxy: scalar parameter  (.value read-only)
# =========================================================
class _Scalar:
    """Mimics a Pyomo scalar Param (read-only, numeric)."""
    __slots__ = ("_v",)

    def __init__(self, v: float) -> None:
        self._v = float(v)

    @property
    def value(self) -> float:
        return self._v

    def __float__(self) -> float:
        return self._v

    def __int__(self) -> int:
        return int(self._v)

    def __index__(self) -> int:
        return int(self._v)

    def __call__(self) -> float:
        return self._v

    def __repr__(self) -> str:
        return f"_Scalar({self._v})"


# =========================================================
# Proxy: indexed parameter  (e.g. a[i] or a[i,t])
# =========================================================
class _IndexProxy:
    """Mimics a Pyomo Param(I) or Param(I, T)."""
    __slots__ = ("_d",)

    def __init__(self, data: dict) -> None:
        self._d = {_nk(k): float(v) for k, v in data.items()}

    def __getitem__(self, key) -> float:
        return self._d[_nk(key)]

    def __call__(self, *args) -> float:
        return self[args[0] if len(args) == 1 else args]


# =========================================================
# Proxy: single GRB variable  (.value R/W, .Start warm-start)
# =========================================================
class _VarProxy:
    """
    Wraps one gurobipy Var.

    Before solve   – reads back whatever was written via .value setter
                     (stored in _manual; also forwarded to var.Start).
    After optimize – reading returns var.X (MIP solution) by default.
    After greedy   – writing sets _use_manual=True so new manual values
                     take precedence over stale var.X.
    """
    __slots__ = ("_var", "_manual", "_use_manual")

    def __init__(self, grb_var) -> None:
        object.__setattr__(self, "_var", grb_var)
        object.__setattr__(self, "_manual", None)
        object.__setattr__(self, "_use_manual", False)

    @property
    def value(self):
        if not self._use_manual:
            try:
                x = self._var.X
                if x is not None:
                    return x
            except (gp.GurobiError, AttributeError):
                pass
        return self._manual

    @value.setter
    def value(self, v) -> None:
        object.__setattr__(self, "_manual", v)
        object.__setattr__(self, "_use_manual", v is not None)
        if v is not None:
            try:
                self._var.Start = float(v)
            except Exception:
                pass

    def __float__(self) -> float:
        v = self.value
        return float(v) if v is not None else 0.0

    def __call__(self):
        v = self.value
        return v if v is not None else 0.0

    def __repr__(self) -> str:
        return f"_VarProxy(manual={self._manual}, use_manual={self._use_manual})"


# =========================================================
# Proxy: dict of GRB variables  (Pyomo Var-compatible)
# =========================================================
class _VarDict:
    """Dict-like container of _VarProxy objects."""

    def __init__(self, var_dict: dict) -> None:
        self._p: dict = {_nk(k): _VarProxy(v) for k, v in var_dict.items()}

    def __getitem__(self, key) -> _VarProxy:
        return self._p[_nk(key)]

    def __setitem__(self, key, v) -> None:
        self._p[_nk(key)].value = v

    def __iter__(self):
        return iter(self._p)

    def __contains__(self, key) -> bool:
        return _nk(key) in self._p

    def __bool__(self) -> bool:
        return bool(self._p)

    def __len__(self) -> int:
        return len(self._p)

    def _reset_use_manual(self) -> None:
        """Called after optimize() so reads return var.X instead of _manual."""
        for proxy in self._p.values():
            object.__setattr__(proxy, "_use_manual", False)


# =========================================================
# GRBModelWrapper — Pyomo ConcreteModel-compatible interface
# =========================================================
class GRBModelWrapper:
    """
    Wraps a gurobipy.Model and exposes the same attribute-access pattern as
    a Pyomo ConcreteModel so that existing code (methods.py, full_eval.py,
    dr.py) works without modification.

    Key compatibility points
    ------------------------
    - m.I / m.J / m.T / m.Arcs   → range / list  (iterable)
    - m.Q.value  / m.U.value      → float via _Scalar
    - m.a[i] / m.a[i,t]          → float via _IndexProxy
    - m.x[j,t].value R/W          → _VarDict / _VarProxy
    - m.y, m.z, m.u               → same
    - m.clone()                   → fresh GRBModelWrapper (same structure)
    - Pyomo's value(m.Q) works    → proxy __call__ / __float__
    """

    def __init__(self, gm: gp.Model, meta: SimpleNamespace) -> None:
        self._gm = gm
        self._meta = meta

        # Sets (Pyomo Set-compatible iterables)
        self.I = range(meta.M)
        self.J = range(meta.N)
        self.Arcs = meta.arcs  # list of (i, j) tuples

        # Only set T for multi-period models (has_time checks hasattr(m,'T'))
        if meta.T > 0:
            self.T = range(meta.T)

        # Parameters
        self.Q = _Scalar(meta.Q)
        self.U = _Scalar(meta.site_ub)
        self.a = _IndexProxy(meta.a)

        # Period budget parameter
        if meta.T > 0:
            self.P = _IndexProxy({t: float(meta.P_T[t]) for t in range(meta.T)})
        else:
            self.P = _Scalar(float(meta.P_single))

        # Variable dicts  — only attach if non-empty (hasattr guards in methods.py)
        self.x = _VarDict(meta.x)
        self.y = _VarDict(meta.y)
        if meta.z:
            self.z = _VarDict(meta.z)
        if meta.u:
            self.u = _VarDict(meta.u)

    # ------------------------------------------------------------------
    # After optimize(): reset proxies so var.X takes precedence
    # ------------------------------------------------------------------
    def _mark_solved(self) -> None:
        """Reset all variable proxies to read from GRB solution (var.X)."""
        for attr in ("x", "y", "z", "u"):
            vd = getattr(self, attr, None)
            if isinstance(vd, _VarDict):
                vd._reset_use_manual()

    # ------------------------------------------------------------------
    # clone(): rebuild structurally identical wrapper with fresh variables
    # ------------------------------------------------------------------
    def clone(self) -> "GRBModelWrapper":
        """
        Create a structurally identical wrapper with zeroed variables.
        Used by full_eval.full_eval_from_U for greedy re-evaluation.
        """
        meta = self._meta
        if meta.T > 0:
            return build_multi_period_model_grb(
                M=meta.M, N=meta.N, T=meta.T,
                in_range=meta.arcs,
                Ji=meta.Ji, Ij=meta.Ij,
                demand_IT=meta.demand_arr,
                Q=meta.Q, P_T=meta.P_T,
                distIJ=meta.distIJ,
                method_name=meta.method_name,
                allow_multi_charger=meta.allow_multi_charger,
                max_chargers_per_site=meta.max_chargers_per_site,
                cumulative_install=meta.cumulative_install,
            )
        else:
            return build_base_model_grb(
                M=meta.M, N=meta.N,
                in_range=meta.arcs,
                Ji=meta.Ji, Ij=meta.Ij,
                demand_I=meta.demand_arr,
                Q=meta.Q, P=meta.P_single,
                distIJ=meta.distIJ,
                method_name=meta.method_name,
                allow_multi_charger=meta.allow_multi_charger,
                max_chargers_per_site=meta.max_chargers_per_site,
            )

    # ------------------------------------------------------------------
    # Pyomo compatibility stubs
    # ------------------------------------------------------------------
    class _SolnMgr:
        def load_from(self, *a, **kw) -> None:
            pass
    solutions = _SolnMgr()

    def __repr__(self) -> str:
        return (f"<GRBModelWrapper M={self._meta.M} "
                f"N={self._meta.N} T={self._meta.T}>")


# =========================================================
# Helpers for model building
# =========================================================
def _site_ub(total_budget, max_chargers_per_site) -> int:
    upper = int(total_budget) if max_chargers_per_site is None else int(max_chargers_per_site)
    return max(0, upper)


def _compute_farther(distIJ, Ji: dict) -> dict:
    """farther_of[(i,j)] → arcs (i,k) where k is farther than j from i."""
    farther_of: dict = {}
    for i, js in Ji.items():
        js_sorted = sorted(list(js), key=lambda j: distIJ[i][j])
        for pos, j in enumerate(js_sorted):
            farther_of[(i, j)] = [(i, k) for k in js_sorted[pos + 1:]]
    return farther_of


def _apply_method_single(gm, method_name, distIJ, arcs, Ji,
                          open_var: dict, y_vars: dict, a: dict) -> None:
    """Apply single-period policy constraints / modified objective."""
    name = str(method_name).lower()
    if name == "closest_only":
        farther_of = _compute_farther(distIJ, Ji)
        for (i, j), farther in farther_of.items():
            if farther:
                gm.addConstr(
                    gp.quicksum(y_vars[ii, jj] for (ii, jj) in farther)
                    <= 1 - open_var[j],
                    name=f"co_{i}_{j}",
                )
    elif name == "closest_priority":
        cov = gp.quicksum(a[i] * y_vars[i, j] for (i, j) in arcs)
        tie = gp.quicksum(
            (1.0 - float(distIJ[i][j])) * y_vars[i, j] for (i, j) in arcs
        )
        gm.setObjective(cov + 1e-3 * tie, GRB.MAXIMIZE)
    elif name == "system_optimum":
        cov = gp.quicksum(a[i] * y_vars[i, j] for (i, j) in arcs)
        dist_t = gp.quicksum(
            float(distIJ[i][j]) * y_vars[i, j] for (i, j) in arcs
        )
        gm.setObjective(cov - 0.1 * dist_t, GRB.MAXIMIZE)


def _apply_method_multi(gm, method_name, distIJ, arcs, Ji,
                         z_jt: dict, y_ijt: dict, a_it: dict,
                         T_range) -> None:
    """Apply multi-period policy constraints / modified objective."""
    name = str(method_name).lower()
    T_list = list(T_range)
    if name == "closest_only":
        farther_of = _compute_farther(distIJ, Ji)
        for t in T_list:
            for (i, j), farther in farther_of.items():
                if not farther:
                    continue
                gm.addConstr(
                    gp.quicksum(y_ijt[ii, jj, t] for (ii, jj) in farther)
                    <= 1 - z_jt[j, t],
                    name=f"co_{i}_{j}_{t}",
                )
    elif name == "closest_priority":
        cov = gp.quicksum(
            a_it[i, t] * y_ijt[i, j, t] for (i, j) in arcs for t in T_list
        )
        tie = gp.quicksum(
            (1.0 - float(distIJ[i][j])) * y_ijt[i, j, t]
            for (i, j) in arcs for t in T_list
        )
        gm.setObjective(cov + 1e-3 * tie, GRB.MAXIMIZE)
    elif name == "system_optimum":
        cov = gp.quicksum(
            a_it[i, t] * y_ijt[i, j, t] for (i, j) in arcs for t in T_list
        )
        dist_t = gp.quicksum(
            float(distIJ[i][j]) * y_ijt[i, j, t]
            for (i, j) in arcs for t in T_list
        )
        gm.setObjective(cov - 0.1 * dist_t, GRB.MAXIMIZE)


# =========================================================
# Public builder: single-period
# =========================================================
def build_base_model_grb(
    M,
    N,
    in_range,
    Ji,
    Ij,
    demand_I,
    Q,
    P,
    distIJ=None,
    method_name="none",
    allow_multi_charger: bool = False,
    max_chargers_per_site: int | None = None,
) -> GRBModelWrapper:
    """
    gurobipy equivalent of model.build_base_model.
    Returns a GRBModelWrapper with Pyomo-compatible attribute interface.
    """
    gm = gp.Model()
    gm.setParam("OutputFlag", 0)

    arcs = list(in_range)
    site_ub = _site_ub(P, max_chargers_per_site)
    a = {i: float(demand_I[i]) for i in range(M)}

    # --- Variables ---
    x_vars: dict = {}
    z_vars: dict = {}

    if allow_multi_charger:
        for j in range(N):
            z_vars[j] = gm.addVar(vtype=GRB.BINARY, name=f"z_{j}")
            x_vars[j] = gm.addVar(vtype=GRB.INTEGER, lb=0, ub=site_ub,
                                   name=f"x_{j}")
    else:
        for j in range(N):
            x_vars[j] = gm.addVar(vtype=GRB.BINARY, name=f"x_{j}")

    y_vars = {
        (i, j): gm.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}")
        for (i, j) in arcs
    }
    gm.update()

    # --- Objective ---
    gm.setObjective(
        gp.quicksum(a[i] * y_vars[i, j] for (i, j) in arcs),
        GRB.MAXIMIZE,
    )

    # --- Constraints ---
    for i in range(M):
        js = Ji.get(i, [])
        if js:
            gm.addConstr(
                gp.quicksum(y_vars[i, j] for j in js) <= 1,
                name=f"do_{i}",
            )

    if allow_multi_charger:
        for (i, j) in arcs:
            gm.addConstr(y_vars[i, j] <= z_vars[j], name=f"ol_{i}_{j}")
        for j in range(N):
            gm.addConstr(
                x_vars[j] <= site_ub * z_vars[j], name=f"cl_{j}"
            )
        for j in range(N):
            reach = Ij.get(j, [])
            if reach:
                gm.addConstr(
                    gp.quicksum(a[i] * y_vars[i, j] for i in reach)
                    <= Q * x_vars[j],
                    name=f"cap_{j}",
                )
        gm.addConstr(
            gp.quicksum(x_vars[j] for j in range(N)) <= P, name="limit"
        )
    else:
        for (i, j) in arcs:
            gm.addConstr(y_vars[i, j] <= x_vars[j], name=f"ol_{i}_{j}")
        for j in range(N):
            reach = Ij.get(j, [])
            if reach:
                gm.addConstr(
                    gp.quicksum(a[i] * y_vars[i, j] for i in reach)
                    <= Q * x_vars[j],
                    name=f"cap_{j}",
                )
        gm.addConstr(
            gp.quicksum(x_vars[j] for j in range(N)) <= P, name="limit"
        )

    # --- Method-specific constraints / objective ---
    name_lc = str(method_name).lower()
    if name_lc not in ("none", "base") and distIJ is not None:
        open_var = z_vars if allow_multi_charger else x_vars
        _apply_method_single(gm, method_name, distIJ, arcs, Ji,
                              open_var, y_vars, a)

    # --- Metadata stored for clone() and proxy building ---
    meta = SimpleNamespace(
        M=M, N=N, T=0,
        arcs=arcs,
        Ji=dict(Ji), Ij=dict(Ij),
        Q=float(Q),
        P_single=int(P), P_T=None,
        site_ub=site_ub,
        allow_multi_charger=allow_multi_charger,
        cumulative_install=False,
        method_name=str(method_name),
        max_chargers_per_site=max_chargers_per_site,
        distIJ=distIJ,
        demand_arr=np.asarray(demand_I, dtype=float),
        x=x_vars, y=y_vars, z=z_vars, u={},
        a=a,
    )

    return GRBModelWrapper(gm, meta)


# =========================================================
# Public builder: multi-period
# =========================================================
def build_multi_period_model_grb(
    M,
    N,
    T,
    in_range,
    Ji,
    Ij,
    demand_IT,
    Q,
    P_T,
    distIJ=None,
    method_name="none",
    allow_multi_charger: bool = True,
    max_chargers_per_site: int | None = None,
    cumulative_install: bool = True,
) -> GRBModelWrapper:
    """
    gurobipy equivalent of model.build_multi_period_model.
    Returns a GRBModelWrapper with Pyomo-compatible attribute interface.
    """
    P_T = list(P_T)
    if len(P_T) != T:
        raise ValueError(f"P_T length must be T={T}, got {len(P_T)}")

    gm = gp.Model()
    gm.setParam("OutputFlag", 0)

    arcs = list(in_range)
    site_ub = _site_ub(max(P_T), max_chargers_per_site)
    demand_IT_arr = np.asarray(demand_IT, dtype=float)  # shape (T, M)

    # demand dict: (i, t) -> float  (same indexing as Pyomo Param(I, T))
    a = {
        (i, t): float(demand_IT_arr[t, i])
        for i in range(M)
        for t in range(T)
    }

    # --- Variables ---
    x_vars: dict = {}
    y_vars: dict = {}
    z_vars: dict = {}
    u_vars: dict = {}

    if not allow_multi_charger:
        for j in range(N):
            for t in range(T):
                x_vars[j, t] = gm.addVar(vtype=GRB.BINARY,
                                          name=f"x_{j}_{t}")
        for (i, j) in arcs:
            for t in range(T):
                y_vars[i, j, t] = gm.addVar(vtype=GRB.BINARY,
                                             name=f"y_{i}_{j}_{t}")
    else:
        for j in range(N):
            for t in range(T):
                u_vars[j, t] = gm.addVar(vtype=GRB.INTEGER, lb=0,
                                          ub=site_ub, name=f"u_{j}_{t}")
                x_vars[j, t] = gm.addVar(vtype=GRB.INTEGER, lb=0,
                                          ub=site_ub, name=f"x_{j}_{t}")
                z_vars[j, t] = gm.addVar(vtype=GRB.BINARY,
                                          name=f"z_{j}_{t}")
        for (i, j) in arcs:
            for t in range(T):
                y_vars[i, j, t] = gm.addVar(vtype=GRB.BINARY,
                                             name=f"y_{i}_{j}_{t}")
    gm.update()

    # --- Objective ---
    gm.setObjective(
        gp.quicksum(
            a[i, t] * y_vars[i, j, t]
            for (i, j) in arcs
            for t in range(T)
        ),
        GRB.MAXIMIZE,
    )

    # --- Demand-once per period ---
    for i in range(M):
        js = Ji.get(i, [])
        if js:
            for t in range(T):
                gm.addConstr(
                    gp.quicksum(y_vars[i, j, t] for j in js) <= 1,
                    name=f"do_{i}_{t}",
                )

    if not allow_multi_charger:
        # open-link, capacity, budget
        for (i, j) in arcs:
            for t in range(T):
                gm.addConstr(
                    y_vars[i, j, t] <= x_vars[j, t],
                    name=f"ol_{i}_{j}_{t}",
                )
        for j in range(N):
            for t in range(T):
                reach = Ij.get(j, [])
                if reach:
                    gm.addConstr(
                        gp.quicksum(a[i, t] * y_vars[i, j, t] for i in reach)
                        <= Q * x_vars[j, t],
                        name=f"cap_{j}_{t}",
                    )
        for t in range(T):
            gm.addConstr(
                gp.quicksum(x_vars[j, t] for j in range(N)) <= P_T[t],
                name=f"lim_{t}",
            )
    else:
        # x >= z  (open-charger link)
        for j in range(N):
            for t in range(T):
                gm.addConstr(
                    x_vars[j, t] >= z_vars[j, t],
                    name=f"oc_{j}_{t}",
                )
        # y <= z  (open link)
        for (i, j) in arcs:
            for t in range(T):
                gm.addConstr(
                    y_vars[i, j, t] <= z_vars[j, t],
                    name=f"ol_{i}_{j}_{t}",
                )
        # x <= U * z  (charger link)
        for j in range(N):
            for t in range(T):
                gm.addConstr(
                    x_vars[j, t] <= site_ub * z_vars[j, t],
                    name=f"cl_{j}_{t}",
                )
        # capacity
        for j in range(N):
            for t in range(T):
                reach = Ij.get(j, [])
                if reach:
                    gm.addConstr(
                        gp.quicksum(
                            a[i, t] * y_vars[i, j, t] for i in reach
                        ) <= Q * x_vars[j, t],
                        name=f"cap_{j}_{t}",
                    )
        # dynamics: x[j,t] = x[j,t-1] + u[j,t]  (or just u[j,t])
        for j in range(N):
            for t in range(T):
                if cumulative_install:
                    if t == 0:
                        gm.addConstr(
                            x_vars[j, t] == u_vars[j, t],
                            name=f"dyn_{j}_{t}",
                        )
                    else:
                        gm.addConstr(
                            x_vars[j, t] == x_vars[j, t - 1] + u_vars[j, t],
                            name=f"dyn_{j}_{t}",
                        )
                else:
                    gm.addConstr(
                        x_vars[j, t] == u_vars[j, t],
                        name=f"dyn_{j}_{t}",
                    )
        # budget per period (on u)
        for t in range(T):
            gm.addConstr(
                gp.quicksum(u_vars[j, t] for j in range(N)) <= P_T[t],
                name=f"lim_{t}",
            )

    # --- Method-specific constraints / objective ---
    name_lc = str(method_name).lower()
    if name_lc not in ("none", "base") and distIJ is not None:
        _apply_method_multi(
            gm, method_name, distIJ, arcs, Ji,
            z_vars, y_vars, a, range(T),
        )

    # --- Metadata ---
    meta = SimpleNamespace(
        M=M, N=N, T=T,
        arcs=arcs,
        Ji=dict(Ji), Ij=dict(Ij),
        Q=float(Q),
        P_single=None, P_T=list(P_T),
        site_ub=site_ub,
        allow_multi_charger=allow_multi_charger,
        cumulative_install=cumulative_install,
        method_name=str(method_name),
        max_chargers_per_site=max_chargers_per_site,
        distIJ=distIJ,
        demand_arr=demand_IT_arr,
        x=x_vars, y=y_vars, z=z_vars, u=u_vars,
        a=a,
    )

    return GRBModelWrapper(gm, meta)
