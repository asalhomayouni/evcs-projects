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
    allow_multi_charger: bool = False,
    max_chargers_per_site: int | None = None,
):
   
    m = ConcreteModel()

    # --- Sets ---
    m.I = Set(initialize=range(M))
    m.J = Set(initialize=range(N))
    m.Arcs = Set(dimen=2, initialize=in_range)

    # --- Parameters ---
    m.a = Param(m.I, initialize={i: float(demand_I[i]) for i in range(M)}, within=NonNegativeReals)

    # Q = capacity per charger/module (in "demand units")
    m.Q = Param(initialize=float(Q), within=NonNegativeReals, mutable=True)

    # P = budget (interpretation depends on allow_multi_charger)
    # keep as integer-like
    m.P = Param(initialize=int(P), within=NonNegativeIntegers, mutable=True)

    # Upper bound on chargers per site (U)
    # If not provided, a safe default is P (can't place more chargers at a site than total budget)
    U = int(P) if max_chargers_per_site is None else int(max_chargers_per_site)
    if U <= 0:
        U = 0
    m.U = Param(initialize=U, within=NonNegativeIntegers, mutable=True)

    # --- Decision Variables ---
    if allow_multi_charger:
        # z[j]=1 if site is used at all, x[j]=# chargers there
        m.z = Var(m.J, within=Binary)
        m.x = Var(m.J, within=NonNegativeIntegers, bounds=(0, U))
    else:
        # x[j]=1 if site open
        m.x = Var(m.J, within=Binary)

    # Binary assignment (each demand chooses at most one station)
    m.y = Var(m.Arcs, within=Binary)


    # --- Objective: maximize covered demand ---
    def obj_rule(m):
        return sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    m.obj = Objective(rule=obj_rule, sense=maximize)

    # --- Constraints ---
    # Each demand point assigned at most once
    def demand_once(m, i):
        return sum(m.y[i, j] for j in Ji.get(i, [])) <= 1
    m.demand_once = Constraint(m.I, rule=demand_once)

    # Link assignment to open site
    if allow_multi_charger:
        def open_link(m, i, j):
            return m.y[i, j] <= m.z[j]
        m.open_link = Constraint(m.Arcs, rule=open_link)

        # Link chargers to open indicator: x[j] <= U*z[j]
        def charger_link(m, j):
            return m.x[j] <= m.U * m.z[j]
        m.charger_link = Constraint(m.J, rule=charger_link)

        # Capacity scales with charger count
        def capacity_rule(m, j):
            return sum(m.a[i] * m.y[i, j] for i in Ij.get(j, [])) <= m.Q * m.x[j]
        m.capacity = Constraint(m.J, rule=capacity_rule)

        # Budget is total chargers/modules
        def limit_rule(m):
            return sum(m.x[j] for j in m.J) <= m.P
        m.limit = Constraint(rule=limit_rule)

    else:
        # Binary case: y <= x
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


from pyomo.environ import ConstraintList  # add at top if not already

def build_multi_period_model(
    M, N, T, in_range, Ji, Ij,
    demand_IT,          # shape: [T][M] or dict {(i,t): val}
    Q, P_T,             # P_T: list/array length T (budget per period)
    distIJ=None, method_name="none",
    allow_multi_charger: bool = True,
    max_chargers_per_site: int | None = None,
    cumulative_install: bool = True,
):
    
    from evcs.methods import apply_method_multi, compute_farther  # local import to avoid circular issues

    m = ConcreteModel()

    # --- Sets ---
    m.I = Set(initialize=range(M))
    m.J = Set(initialize=range(N))
    m.T = Set(initialize=range(T))
    m.Arcs = Set(dimen=2, initialize=in_range)

    # --- Parameters ---
    # Demand a[i,t]
    if isinstance(demand_IT, dict):
        a_init = {(i, t): float(demand_IT[(i, t)]) for t in range(T) for i in range(M)}
    else:
        # assume list/np array like demand_IT[t][i]
        a_init = {(i, t): float(demand_IT[t][i]) for t in range(T) for i in range(M)}
    m.a = Param(m.I, m.T, initialize=a_init, within=NonNegativeReals)

    m.Q = Param(initialize=float(Q), within=NonNegativeReals, mutable=True)

    # Period budgets
    if not hasattr(P_T, "__len__"):
        raise ValueError("P_T must be a list/array of length T (budget per period).")
    if len(P_T) != T:
        raise ValueError(f"P_T length must be T={T}, got {len(P_T)}")
    m.P = Param(m.T, initialize={t: int(P_T[t]) for t in range(T)}, within=NonNegativeIntegers, mutable=True)

    # Upper bound per site
    U = int(max(P_T)) if max_chargers_per_site is None else int(max_chargers_per_site)
    if U < 0:
        U = 0
    m.U = Param(initialize=U, within=NonNegativeIntegers, mutable=True)

    # --- Decision variables ---
    # Recommend multi-charger for multi-period
    if not allow_multi_charger:
        # You CAN model binary stations per period, but it’s usually not what you want for installation planning.
        # Kept here for completeness.
        m.x = Var(m.J, m.T, within=Binary)
        m.y = Var(m.Arcs, m.T, within=Binary)

    else:
        m.u = Var(m.J, m.T, within=NonNegativeIntegers, bounds=(0, U))  # new installs
        m.x = Var(m.J, m.T, within=NonNegativeIntegers, bounds=(0, U))  # installed/available
        m.z = Var(m.J, m.T, within=Binary)                               # open indicator
        m.y = Var(m.Arcs, m.T, within=Binary)


    # --- Objective: maximize total covered demand (sum over time) ---
    def obj_rule(m):
        return sum(m.a[i, t] * m.y[i, j, t] for (i, j) in m.Arcs for t in m.T)
    m.obj = Objective(rule=obj_rule, sense=maximize)

    # --- Constraints ---
    # Each demand point assigned at most once per period
    def demand_once(m, i, t):
        return sum(m.y[i, j, t] for j in Ji.get(i, [])) <= 1
    m.demand_once = Constraint(m.I, m.T, rule=demand_once)

    if not allow_multi_charger:
        # Binary-per-period case
        def open_link(m, i, j, t):
            return m.y[i, j, t] <= m.x[j, t]
        m.open_link = Constraint(m.Arcs, m.T, rule=open_link)

        def capacity_rule(m, j, t):
            return sum(m.a[i, t] * m.y[i, j, t] for i in Ij.get(j, [])) <= m.Q * m.x[j, t]
        m.capacity = Constraint(m.J, m.T, rule=capacity_rule)

        # Budget each period (how many sites open in each period)
        def limit_rule(m, t):
            return sum(m.x[j, t] for j in m.J) <= m.P[t]
        m.limit = Constraint(m.T, rule=limit_rule)

    else:
        # Link assignment to open site
        def open_link(m, i, j, t):
            return m.y[i, j, t] <= m.z[j, t]
        m.open_link = Constraint(m.Arcs, m.T, rule=open_link)

        # Link chargers to open indicator
        def charger_link(m, j, t):
            return m.x[j, t] <= m.U * m.z[j, t]
        m.charger_link = Constraint(m.J, m.T, rule=charger_link)

        # Capacity scales with chargers in that period
        def capacity_rule(m, j, t):
            return sum(m.a[i, t] * m.y[i, j, t] for i in Ij.get(j, [])) <= m.Q * m.x[j, t]
        m.capacity = Constraint(m.J, m.T, rule=capacity_rule)

        # Installation dynamics
        if cumulative_install:
            def dyn(m, j, t):
                if t == 0:
                    return m.x[j, t] == m.u[j, t]
                return m.x[j, t] == m.x[j, t-1] + m.u[j, t]
            m.dyn = Constraint(m.J, m.T, rule=dyn)
        else:
            # If you want independent x each period (no persistence), force x=u
            def dyn(m, j, t):
                return m.x[j, t] == m.u[j, t]
            m.dyn = Constraint(m.J, m.T, rule=dyn)

        # Budget is installs per period
        def limit_rule(m, t):
            return sum(m.u[j, t] for j in m.J) <= m.P[t]
        m.limit = Constraint(m.T, rule=limit_rule)

    # --- Apply policy-specific logic (multi-period aware) ---
    if str(method_name).lower() not in ["none", "base"]:
        farther_of = compute_farther(distIJ, in_range, Ji) if distIJ is not None else {}
        m = apply_method_multi(m, method_name, distIJ, in_range, Ji, Ij, farther_of, verbose=False)

    return m
