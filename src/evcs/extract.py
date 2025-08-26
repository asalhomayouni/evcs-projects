from pyomo.environ import value

def _v(x):
    try: return float(value(x))
    except Exception: return 0.0

def fractions_by_node(m, in_range):
    """Returns dict i -> fraction served (sum_j y[i,j])."""
    frac = {i: 0.0 for i in m.I}
    for (i, j) in in_range:
        y = _v(m.y[i, j])
        if y > 1e-9:
            frac[i] += y
   ######
    for i in frac:
        frac[i] = max(0.0, min(1.0, frac[i]))
    return frac

def extract_solution(m, in_range):
    """opened_J (indices in J-space), best_of (i->(j,y)), frac dict, covered amount."""
    opened_J = [j for j in m.J if _v(m.x[j]) > 0.5]

    best_of = {}
    for (i, j) in in_range:
        y = _v(m.y[i, j])
        if y > 1e-9:
            if (i not in best_of) or (y > best_of[i][1]):
                best_of[i] = (j, y)

    frac = fractions_by_node(m, in_range)
    covered = _v(m.obj)
    return dict(opened_J=opened_J, best_of=best_of, frac=frac, covered=covered)
