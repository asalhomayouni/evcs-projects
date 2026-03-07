# src/evcs/model.py
from __future__ import annotations

from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeIntegers,
    NonNegativeReals,
    Objective,
    Param,
    Set,
    Var,
    maximize,
)

from evcs.methods import apply_method, apply_method_multi, compute_farther


def _init_site_capacity_upper_bound(total_budget, max_chargers_per_site: int | None) -> int:
    """Shared upper-bound logic for charger/site variables."""
    upper = int(total_budget) if max_chargers_per_site is None else int(max_chargers_per_site)
    return max(0, upper)


def _build_single_period_demand_init(M, demand_I):
    return {i: float(demand_I[i]) for i in range(M)}


def _build_multi_period_demand_init(M, T, demand_IT):
    if isinstance(demand_IT, dict):
        return {(i, t): float(demand_IT[(i, t)]) for t in range(T) for i in range(M)}
    return {(i, t): float(demand_IT[t][i]) for t in range(T) for i in range(M)}



def build_base_model(
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
):
    m = ConcreteModel()

    # --- Sets ---
    m.I = Set(initialize=range(M))
    m.J = Set(initialize=range(N))
    m.Arcs = Set(dimen=2, initialize=in_range)

    # --- Parameters ---
    m.a = Param(
        m.I,
        initialize=_build_single_period_demand_init(M, demand_I),
        within=NonNegativeReals,
    )
    m.Q = Param(initialize=float(Q), within=NonNegativeReals, mutable=True)
    m.P = Param(initialize=int(P), within=NonNegativeIntegers, mutable=True)

    site_ub = _init_site_capacity_upper_bound(P, max_chargers_per_site)
    m.U = Param(initialize=site_ub, within=NonNegativeIntegers, mutable=True)

    # --- Decision Variables ---
    if allow_multi_charger:
        m.z = Var(m.J, within=Binary)
        m.x = Var(m.J, within=NonNegativeIntegers, bounds=(0, site_ub))
    else:
        m.x = Var(m.J, within=Binary)

    m.y = Var(m.Arcs, within=Binary)

    # --- Objective ---
    def obj_rule(m):
        return sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)

    m.obj = Objective(rule=obj_rule, sense=maximize)

    # --- Constraints ---
    def demand_once(m, i):
        return sum(m.y[i, j] for j in Ji.get(i, [])) <= 1

    m.demand_once = Constraint(m.I, rule=demand_once)

    if allow_multi_charger:
        def open_link(m, i, j):
            return m.y[i, j] <= m.z[j]

        def charger_link(m, j):
            return m.x[j] <= m.U * m.z[j]

        def capacity_rule(m, j):
            return sum(m.a[i] * m.y[i, j] for i in Ij.get(j, [])) <= m.Q * m.x[j]

        def limit_rule(m):
            return sum(m.x[j] for j in m.J) <= m.P

        m.open_link = Constraint(m.Arcs, rule=open_link)
        m.charger_link = Constraint(m.J, rule=charger_link)
        m.capacity = Constraint(m.J, rule=capacity_rule)
        m.limit = Constraint(rule=limit_rule)
    else:
        def open_link(m, i, j):
            return m.y[i, j] <= m.x[j]

        def capacity_rule(m, j):
            return sum(m.a[i] * m.y[i, j] for i in Ij.get(j, [])) <= m.Q * m.x[j]

        def limit_rule(m):
            return sum(m.x[j] for j in m.J) <= m.P

        m.open_link = Constraint(m.Arcs, rule=open_link)
        m.capacity = Constraint(m.J, rule=capacity_rule)
        m.limit = Constraint(rule=limit_rule)

    if str(method_name).lower() not in ["none", "base"]:
        farther_of = compute_farther(distIJ, in_range, Ji) if distIJ is not None else {}
        m = apply_method(m, method_name, distIJ, in_range, Ji, Ij, farther_of, verbose=False)

    return m



def build_multi_period_model(
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
):
    m = ConcreteModel()

    # --- Sets ---
    m.I = Set(initialize=range(M))
    m.J = Set(initialize=range(N))
    m.T = Set(initialize=range(T))
    m.Arcs = Set(dimen=2, initialize=in_range)

    # --- Parameters ---
    m.a = Param(
        m.I,
        m.T,
        initialize=_build_multi_period_demand_init(M, T, demand_IT),
        within=NonNegativeReals,
    )
    m.Q = Param(initialize=float(Q), within=NonNegativeReals, mutable=True)

    if not hasattr(P_T, "__len__"):
        raise ValueError("P_T must be a list/array of length T (budget per period).")
    if len(P_T) != T:
        raise ValueError(f"P_T length must be T={T}, got {len(P_T)}")

    m.P = Param(
        m.T,
        initialize={t: int(P_T[t]) for t in range(T)},
        within=NonNegativeIntegers,
        mutable=True,
    )

    site_ub = _init_site_capacity_upper_bound(max(P_T), max_chargers_per_site)
    m.U = Param(initialize=site_ub, within=NonNegativeIntegers, mutable=True)

    # --- Decision variables ---
    if not allow_multi_charger:
        m.x = Var(m.J, m.T, within=Binary)
        m.y = Var(m.Arcs, m.T, within=Binary)
    else:
        m.u = Var(m.J, m.T, within=NonNegativeIntegers, bounds=(0, site_ub))
        m.x = Var(m.J, m.T, within=NonNegativeIntegers, bounds=(0, site_ub))
        m.z = Var(m.J, m.T, within=Binary)
        m.y = Var(m.Arcs, m.T, within=Binary)

        def open_charger_link(m, j, t):
            return m.x[j, t] >= m.z[j, t]

        m.open_charger_link = Constraint(m.J, m.T, rule=open_charger_link)

    # --- Objective ---
    def obj_rule(m):
        return sum(m.a[i, t] * m.y[i, j, t] for (i, j) in m.Arcs for t in m.T)

    m.obj = Objective(rule=obj_rule, sense=maximize)

    # --- Constraints ---
    def demand_once(m, i, t):
        return sum(m.y[i, j, t] for j in Ji.get(i, [])) <= 1

    m.demand_once = Constraint(m.I, m.T, rule=demand_once)

    if not allow_multi_charger:
        def open_link(m, i, j, t):
            return m.y[i, j, t] <= m.x[j, t]

        def capacity_rule(m, j, t):
            return sum(m.a[i, t] * m.y[i, j, t] for i in Ij.get(j, [])) <= m.Q * m.x[j, t]

        def limit_rule(m, t):
            return sum(m.x[j, t] for j in m.J) <= m.P[t]

        m.open_link = Constraint(m.Arcs, m.T, rule=open_link)
        m.capacity = Constraint(m.J, m.T, rule=capacity_rule)
        m.limit = Constraint(m.T, rule=limit_rule)
    else:
        def open_link(m, i, j, t):
            return m.y[i, j, t] <= m.z[j, t]

        def charger_link(m, j, t):
            return m.x[j, t] <= m.U * m.z[j, t]

        def capacity_rule(m, j, t):
            return sum(m.a[i, t] * m.y[i, j, t] for i in Ij.get(j, [])) <= m.Q * m.x[j, t]

        def dyn(m, j, t):
            if cumulative_install:
                if t == 0:
                    return m.x[j, t] == m.u[j, t]
                return m.x[j, t] == m.x[j, t - 1] + m.u[j, t]
            return m.x[j, t] == m.u[j, t]

        def limit_rule(m, t):
            return sum(m.u[j, t] for j in m.J) <= m.P[t]

        m.open_link = Constraint(m.Arcs, m.T, rule=open_link)
        m.charger_link = Constraint(m.J, m.T, rule=charger_link)
        m.capacity = Constraint(m.J, m.T, rule=capacity_rule)
        m.dyn = Constraint(m.J, m.T, rule=dyn)
        m.limit = Constraint(m.T, rule=limit_rule)

    if str(method_name).lower() not in ["none", "base"]:
        farther_of = compute_farther(distIJ, in_range, Ji) if distIJ is not None else {}
        m = apply_method_multi(m, method_name, distIJ, in_range, Ji, Ij, farther_of, verbose=False)

    return m
