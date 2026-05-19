"""
Bipartite assignment solvers for EVCS demand-to-station sub-problems.

Why this exists
---------------
The System Optimum (SO) assignment problem — given fixed station placements,
assign demand zones to stations to maximise coverage − λ·distance — is a
bipartite transportation LP, NOT a MIP.  The original code declared y[i,j]
as Binary and sent the whole thing to Gurobi branch-and-bound, which is
orders of magnitude slower than necessary.

Two solvers are provided:

* solve_max_coverage_flow  – pure max-coverage (λ=0) via
  scipy.sparse.csgraph.maximum_flow (integer push-relabel, very fast).

* solve_so_assignment_lp   – SO objective (coverage − λ·distance) as a
  sparse LP via scipy.optimize.linprog (HiGHS backend).  The constraint
  matrix is the bipartite node-arc incidence matrix; when a[i]≡1 it is
  totally unimodular so the LP optimum is always integral.  For varying
  a[i] the LP is still the correct continuous relaxation of the
  transportation problem.

Both functions are pure (no side effects on model objects) and accept the
same arc / demand / capacity dicts so callers can swap them freely.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Max-coverage: bipartite max-flow via scipy.sparse.csgraph
# ---------------------------------------------------------------------------

def solve_max_coverage_flow(
    arcs: list[tuple[int, int]],
    a: dict[int, float],
    cap: dict[int, float],
    scale: int = 1000,
) -> dict[tuple[int, int], float]:
    """
    Maximum demand coverage as an integer bipartite max-flow.

    Graph layout (node ids)::

        0 (source)
        1 … M          demand zone nodes  i
        M+1 … M+N      station nodes      j
        M+N+1 (sink)

    Edges::

        source → i   : capacity = round(a[i] * scale)
        i      → j   : capacity = round(a[i] * scale)   for each (i,j) in arcs
        j      → sink : capacity = round(cap[j] * scale)

    Parameters
    ----------
    arcs  : list of (i, j) pairs — all arcs to *open* stations
    a     : demand per zone  {i: float}
    cap   : station capacity {j: float}  (Q·x[j] for open stations)
    scale : multiplier to convert float→int (default 1000)

    Returns
    -------
    y : {(i, j): float} values in [0, 1]
        1.0 means zone i is fully routed to station j.
    """
    from scipy.sparse.csgraph import maximum_flow

    if not arcs:
        return {}

    I_nodes = sorted({i for i, _ in arcs})
    J_nodes = sorted({j for _, j in arcs})
    M, N = len(I_nodes), len(J_nodes)
    n = M + N + 2
    s, t = 0, n - 1
    i_idx = {node: 1 + k for k, node in enumerate(I_nodes)}
    j_idx = {node: 1 + M + k for k, node in enumerate(J_nodes)}

    rows, cols, data = [], [], []

    # source → demand_i
    for i in I_nodes:
        ai = max(0, int(round(float(a.get(i, 0.0)) * scale)))
        if ai > 0:
            rows.append(s)
            cols.append(i_idx[i])
            data.append(ai)

    # demand_i → station_j
    for (i, j) in arcs:
        ai = max(0, int(round(float(a.get(i, 0.0)) * scale)))
        if ai > 0:
            rows.append(i_idx[i])
            cols.append(j_idx[j])
            data.append(ai)

    # station_j → sink
    for j in J_nodes:
        cj = max(0, int(round(float(cap.get(j, 0.0)) * scale)))
        if cj > 0:
            rows.append(j_idx[j])
            cols.append(t)
            data.append(cj)

    if not data:
        return {(i, j): 0.0 for (i, j) in arcs}

    C = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int64)
    result = maximum_flow(C, s, t)
    flow = result.flow

    y: dict[tuple[int, int], float] = {}
    for (i, j) in arcs:
        ai_scaled = max(1, int(round(float(a.get(i, 0.0)) * scale)))
        f = float(flow[i_idx[i], j_idx[j]])
        y[i, j] = min(1.0, f / ai_scaled)
    return y


# ---------------------------------------------------------------------------
# SO assignment: sparse LP via scipy HiGHS
# ---------------------------------------------------------------------------

def solve_so_assignment_lp(
    arcs: list[tuple[int, int]],
    a: dict[int, float],
    cap: dict[int, float],
    dij_fn,
    lambda_dist: float = 0.1,
) -> dict[tuple[int, int], float]:
    """
    System Optimum assignment as a bipartite min-cost LP (HiGHS backend).

    Maximises::

        Σ_{(i,j)} (a[i] − λ·dist(i,j)) · y[i,j]

    Subject to::

        Σ_j y[i,j]       ≤ 1       ∀ i   (each demand served at most once)
        Σ_i a[i]·y[i,j]  ≤ cap[j]  ∀ j   (station capacity)
        0 ≤ y[i,j] ≤ 1

    The sparse constraint matrix is built with scipy.sparse so that HiGHS
    can exploit it without any dense-matrix overhead.

    Parameters
    ----------
    arcs       : list of (i, j) pairs — open arcs only
    a          : demand per zone {i: float}
    cap        : station capacity {j: float}  (Q·x[j], open stations only)
    dij_fn     : callable (i, j) → float  distance between zone i and station j
    lambda_dist: distance penalty weight (default 0.1)

    Returns
    -------
    y : {(i, j): float} LP-optimal assignment values in [0, 1].
        Under total unimodularity (unit demands) these are exactly 0 or 1.
    """
    from scipy.optimize import linprog

    if not arcs:
        return {}

    K = len(arcs)
    I_nodes = sorted({i for i, _ in arcs})
    J_nodes = sorted({j for _, j in arcs})

    # Objective: minimise Σ (λ·dist − a[i]) · y  (≡ maximise SO profit)
    c_obj = np.array(
        [lambda_dist * float(dij_fn(i, j)) - float(a.get(i, 0.0)) for (i, j) in arcs],
        dtype=float,
    )

    row, col, val, b = [], [], [], []
    r = 0

    # Constraint 1: Σ_j y[i,j] ≤ 1  for each demand zone i
    for i in I_nodes:
        for k, (ii, _jj) in enumerate(arcs):
            if ii == i:
                row.append(r)
                col.append(k)
                val.append(1.0)
        b.append(1.0)
        r += 1

    # Constraint 2: Σ_i a[i]·y[i,j] ≤ cap[j]  for each station j
    for j in J_nodes:
        for k, (_ii, jj) in enumerate(arcs):
            if jj == j:
                row.append(r)
                col.append(k)
                val.append(float(a.get(_ii, 0.0)))
        b.append(float(cap.get(j, 0.0)))
        r += 1

    A_ub = csr_matrix((val, (row, col)), shape=(r, K))

    result = linprog(
        c_obj,
        A_ub=A_ub,
        b_ub=np.asarray(b, dtype=float),
        bounds=[(0.0, 1.0)] * K,
        method="highs",
    )

    if result.status != 0:
        return {(i, j): 0.0 for (i, j) in arcs}

    return {(i, j): float(result.x[k]) for k, (i, j) in enumerate(arcs)}
