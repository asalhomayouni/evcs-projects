# src/evcs/solve.py
import os
from pyomo.contrib.appsi.solvers import Highs
from pyomo.opt import TerminationCondition

def solve_model(
    m,
    time_limit=300,     # seconds
    mip_gap=0.01,       # relative MIP gap
    threads=None,       # default: all cores - 1
    presolve=True,
    verbose=False,
    load_solution=True,  # ✅ NEW: control auto-loading
):
    opt = Highs()

    # Configure solver options safely
    def safe_set(attr, value):
        try:
            setattr(opt.config, attr, value)
        except Exception:
            pass

    # stream output
    safe_set("stream_solver", bool(verbose))

    if threads is None:
        threads = max(1, (os.cpu_count() or 2) - 1)

    # Apply solver options
    safe_set("time_limit", float(time_limit))
    safe_set("mip_rel_gap", float(mip_gap))
    safe_set("threads", int(threads))
    safe_set("presolve", "on" if presolve else "off")

    # ✅ IMPORTANT: prevent "feasible but suboptimal" auto-load warning
    safe_set("load_solution", bool(load_solution))

    # Solve model
    res = opt.solve(m)

    # Optional: load into model
    if load_solution:
        try:
            if hasattr(res, "solution_loader") and res.solution_loader is not None:
                res.solution_loader.load_vars()
        except Exception:
            pass

    return res


