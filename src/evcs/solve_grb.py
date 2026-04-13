# src/evcs/solve_grb.py
"""
Native gurobipy solver wrapper.

solve_model_grb(wrapper, ...)  calls gm.optimize() directly — no Pyomo
bridge — and returns the same SimpleNamespace interface as solve.solve_model
so callers need no changes.

Termination strings are mapped to the same values that the Pyomo/Gurobi path
produced, so existing checks like
    res.termination_condition == TerminationCondition.optimal
continue to work (TerminationCondition.optimal == "optimal").
"""
from __future__ import annotations

from types import SimpleNamespace

import gurobipy as gp
from gurobipy import GRB

from evcs.model_grb import GRBModelWrapper


# Map Gurobi status codes → strings matching Pyomo TerminationCondition names
_STATUS_MAP = {
    GRB.OPTIMAL:        "optimal",
    GRB.TIME_LIMIT:     "maxTimeLimit",
    GRB.INFEASIBLE:     "infeasible",
    GRB.INF_OR_UNBD:    "infeasibleOrUnbounded",
    GRB.UNBOUNDED:      "unbounded",
    GRB.NODE_LIMIT:     "maxIterations",
    GRB.SOLUTION_LIMIT: "other",
    GRB.INTERRUPTED:    "other",
    GRB.NUMERIC:        "other",
}


def solve_model_grb(
    wrapper: GRBModelWrapper,
    time_limit: float = 300,
    mip_gap: float = 0.01,
    threads: int | None = None,
    presolve: bool = True,
    verbose: bool = False,
    load_solution: bool = True,
) -> SimpleNamespace:
    """
    Solve a GRBModelWrapper natively via gurobipy.

    Parameters
    ----------
    wrapper       : GRBModelWrapper returned by build_*_model_grb()
    time_limit    : solver wall-clock limit in seconds
    mip_gap       : relative MIP gap tolerance
    threads       : CPU threads (None → 2, matching SLURM default)
    presolve      : enable Gurobi presolve (level 2)
    verbose       : stream solver log to stdout
    load_solution : if True, reset proxies so .value returns MIP solution

    Returns
    -------
    SimpleNamespace with fields:
        .termination_condition    str  ("optimal", "maxTimeLimit", …)
        .best_objective_bound     float or None
        .best_feasible_objective  float or None
        .gurobi_status            int   (raw GRB status code)
        .raw_results              None  (kept for interface compat)
    """
    if not isinstance(wrapper, GRBModelWrapper):
        raise TypeError(
            f"solve_model_grb expects GRBModelWrapper, got {type(wrapper).__name__}. "
            "Use model_grb.build_*_model_grb() to construct the model."
        )

    gm = wrapper._gm
    n_threads = 2 if threads is None else int(threads)

    gm.setParam("OutputFlag", 1 if verbose else 0)
    gm.setParam("TimeLimit", float(time_limit))
    gm.setParam("MIPGap", float(mip_gap))
    gm.setParam("Threads", n_threads)
    gm.setParam("Presolve", 2 if presolve else 0)

    gm.optimize()

    status = gm.Status
    best_obj: float | None = None
    best_bound: float | None = None

    if gm.SolCount > 0:
        try:
            best_obj = float(gm.ObjVal)
        except Exception:
            pass

    try:
        best_bound = float(gm.ObjBound)
    except Exception:
        pass

    term = _STATUS_MAP.get(status, str(status))

    # Let variable proxies read from var.X on next access
    if load_solution and gm.SolCount > 0:
        wrapper._mark_solved()

    return SimpleNamespace(
        termination_condition=term,
        best_objective_bound=best_bound,
        best_feasible_objective=best_obj,
        gurobi_status=status,
        raw_results=None,
    )
