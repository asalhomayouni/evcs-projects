from __future__ import annotations
from types import SimpleNamespace
import gurobipy as gp
from gurobipy import GRB
from evcs.model_grb import GRBModelWrapper

# --- status map ---
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


# --- solver ---
def solve_model_grb(
    wrapper: GRBModelWrapper,
    time_limit: float = 300,
    mip_gap: float = 0.01,
    threads: int | None = None,
    presolve: bool = True,
    verbose: bool = False,
    load_solution: bool = True,
) -> SimpleNamespace:
    if not isinstance(wrapper, GRBModelWrapper):
        raise TypeError(f"expected GRBModelWrapper, got {type(wrapper).__name__}")

    gm = wrapper._gm
    gm.setParam("OutputFlag", 1 if verbose else 0)
    gm.setParam("TimeLimit", float(time_limit))
    gm.setParam("MIPGap", float(mip_gap))
    gm.setParam("Threads", 2 if threads is None else int(threads))
    gm.setParam("Presolve", 2 if presolve else 0)
    gm.optimize()

    status = gm.Status
    best_obj = best_bound = None
    if gm.SolCount > 0:
        try: best_obj = float(gm.ObjVal)
        except Exception: pass
    try: best_bound = float(gm.ObjBound)
    except Exception: pass

    if load_solution and gm.SolCount > 0:
        wrapper._mark_solved()

    return SimpleNamespace(
        termination_condition=_STATUS_MAP.get(status, str(status)),
        best_objective_bound=best_bound,
        best_feasible_objective=best_obj,
        gurobi_status=status,
        raw_results=None,
    )
