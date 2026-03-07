# src/evcs/solve.py
from __future__ import annotations

import os

from pyomo.contrib.appsi.solvers.highs import Highs


def _safe_set_config(opt: Highs, attr: str, value) -> None:
    """Best-effort setter for APPsi HiGHS config fields."""
    try:
        setattr(opt.config, attr, value)
    except Exception:
        pass


def solve_model(
    m,
    time_limit=300,
    mip_gap=0.01,
    threads=None,
    presolve=True,
    verbose=False,
    load_solution=True,
):
    """Solve a Pyomo model with APPsi HiGHS.

    Keeps the public signature unchanged for compatibility with the rest of the
    project skeleton.
    """
    opt = Highs()

    n_threads = threads
    if n_threads is None:
        n_threads = max(1, (os.cpu_count() or 2) - 1)

    _safe_set_config(opt, "stream_solver", bool(verbose))
    _safe_set_config(opt, "time_limit", float(time_limit))
    _safe_set_config(opt, "mip_rel_gap", float(mip_gap))
    _safe_set_config(opt, "threads", int(n_threads))
    _safe_set_config(opt, "presolve", "on" if presolve else "off")

    # Do not auto-load blindly. We only load if a feasible incumbent exists.
    _safe_set_config(opt, "load_solution", False)

    res = opt.solve(m)

    if load_solution:
        try:
            has_feasible = (
                hasattr(res, "best_feasible_objective")
                and res.best_feasible_objective is not None
            )
            if has_feasible and hasattr(res, "solution_loader") and res.solution_loader is not None:
                res.solution_loader.load_vars()
        except Exception:
            pass

    return res
