from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Binary,
    Objective, Constraint, maximize
)

def build_base_model(M, N, in_range, Ji, Ij, demand_I, Q, P):
    """Base p-median-like model with capacity and at-most-one assignment per demand node."""
    m = ConcreteModel()
    m.I = Set(initialize=range(M))
    m.J = Set(initialize=range(N))
    m.Arcs = Set(dimen=2, initialize=in_range)

    m.a = Param(m.I, initialize={i: float(demand_I[i]) for i in range(M)}, within=NonNegativeReals)
    m.Q = Param(initialize=float(Q), within=NonNegativeReals, mutable=True)
    m.P = Param(initialize=int(P), within=NonNegativeReals, mutable=True)

    m.x = Var(m.J, within=Binary)     # open site j?
    m.y = Var(m.Arcs, bounds=(0, 1))  # fraction of demand i assign to j

    # obj: maximize covered demand
    def obj_rule(m):
        return sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    m.obj = Objective(rule=obj_rule, sense=maximize)

    # each demand covered by at most 1
    def cover_once(m, i): return sum(m.y[i, j] for j in Ji.get(i, [])) <= 1
    m.cover_once = Constraint(m.I, rule=cover_once)

    # flow only to open sites
    def open_link(m, i, j): return m.y[i, j] <= m.x[j]
    m.open_link = Constraint(m.Arcs, rule=open_link)

    # capacity per site
    def capacity_rule(m, j): return sum(m.a[i] * m.y[i, j] for i in Ij.get(j, [])) <= m.Q * m.x[j]
    m.capacity = Constraint(m.J, rule=capacity_rule)

    # open at most P sites
    def limit_rule(m): return sum(m.x[j] for j in m.J) <= m.P
    m.limit = Constraint(rule=limit_rule)

    return m
