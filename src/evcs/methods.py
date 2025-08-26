from pyomo.environ import ConstraintList, Objective, maximize

#closest only method
def add_closest_only(m, farther_of):
    """
    'Closest-only' cuts:
    For each (i,j), if j is open, i cannot send flow to stations strictly farther than j.
    sum_{jp farther than j} y[i,jp] <= 1 - x[j]
    """
    m.closest_only = ConstraintList()
    for (i, j), farther in farther_of.items():
        m.closest_only.add(sum(m.y[i, jp] for jp in farther) <= 1 - m.x[j])
    return m

def add_closest_priority(m, weights_J, eps=1e-6):
    """
    Optional alternative: keep same constraints but tweak the objective with a tiny
    tie-breaker that prefers certain sites via weights_J (len N).
    New objective = covered_demand + eps * sum(weights_J[j] * sum_i y[i,j])
    """
    m.obj.deactivate()
    expr_covered = sum(m.a[i] * m.y[i, j] for (i, j) in m.Arcs)
    expr_tiebreak = sum(weights_J[j] * m.y[i, j] for (i, j) in m.Arcs)
    m.obj = Objective(expr=expr_covered + eps * expr_tiebreak, sense=maximize)
    return m
