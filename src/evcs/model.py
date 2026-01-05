# src/evcs/model.py
from __future__ import annotations

from pyomo.environ import (
    ConcreteModel, Set, Param, Var,
    NonNegativeReals, NonNegativeIntegers, Binary,
    Objective, Constraint, maximize
)

from evcs.methods import apply_method, compute_farther


def build_base_model(
    M, N, in_range, Ji, Ij, demand_I, Q, P,
    distIJ=None, method_name="none",
    allow_multi_charger: bool = False
):
    """
    EVCS allocation model.

    If allow_multi_charger=False:
      x[j] is Binary, limit: sum x <= P   (P = number of stations)

    If allow_multi_charger=True:
      x[j] is NonNegativeIntegers, limit: sum x <= P (P = total modules/chargers)
      y <= x still works because x>=1 means open.
      capacity: sum demand*y <= Q*x
    """
    m = ConcreteModel()

    # --- Sets ---
    m.I = Set(initialize=range(M))
    m.J = Set(initialize=range(N))
    m.Arcs = Set(dimen=2, initialize=in_range)

    # --- Parameters ---
    m.a = Param(m.I, initialize={i: float(demand_I[i]) for i in range(M)}, within=NonNegativeReals)
    m.Q = Param(initialize=float(Q), within=NonNegativeReals, mutable=True)
    m.P = Param(initialize=int(P), within=NonNegativeReals, mutable=True)

    # --- Decision Variables ---
    if allow_multi_charger:
        m.x = Var(m.J, within=NonNegativeIntegers)   # number of chargers/modules at j
    else:
        m.x = Var(m.J, within=Binary)

    m.y = Var(m.Arcs, bounds=(0, 1))

    # --- Default Objective: maximize coverage ---
    def obj_rule(m):
        return sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    m.obj = Objective(rule=obj_rule, sense=maximize)

    # --- Constraints ---
    def demand_once(m, i):
        return sum(m.y[i, j] for j in Ji.get(i, [])) <= 1
    m.demand_once = Constraint(m.I, rule=demand_once)

    def open_link(m, i, j):
        return m.y[i, j] <= m.x[j]
    m.open_link = Constraint(m.Arcs, rule=open_link)

    def capacity_rule(m, j):
        return sum(m.a[i] * m.y[i, j] for i in Ij.get(j, [])) <= m.Q * m.x[j]
    m.capacity = Constraint(m.J, rule=capacity_rule)

    def limit_rule(m):
        return sum(m.x[j] for j in m.J) <= m.P
    m.limit = Constraint(rule=limit_rule)

    # --- Apply policy-specific logic ---
    if str(method_name).lower() not in ["none", "base"]:
        farther_of = compute_farther(distIJ, in_range, Ji) if distIJ is not None else {}
        m = apply_method(m, method_name, distIJ, in_range, Ji, Ij, farther_of, verbose=False)

    return m
