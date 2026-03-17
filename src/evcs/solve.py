# src/evcs/solve.py
from __future__ import annotations

import os
from types import SimpleNamespace

from pyomo.environ import SolverFactory, maximize, minimize, value
from pyomo.opt import TerminationCondition

try:
    from pyomo.contrib.appsi.solvers.highs import Highs
except Exception:
    Highs = None


def _safe_float(x):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _extract_best_bound(results, m):
    
    try:
        sense = m.obj.sense
    except Exception:
        sense = maximize

    try:
        lb = _safe_float(results.problem.lower_bound)
    except Exception:
        lb = None

    try:
        ub = _safe_float(results.problem.upper_bound)
    except Exception:
        ub = None

    if sense == maximize:
        return ub
    return lb


def _load_solution_if_available(m, results):
    """
    Load incumbent solution if Pyomo returned one.
    """
    try:
        if hasattr(results, "solution") and results.solution and len(results.solution) > 0:
            m.solutions.load_from(results)
            return True
    except Exception:
        pass

    try:
        m.solutions.load_from(results)
        return True
    except Exception:
        return False


def _solve_with_gurobi(
    m,
    time_limit=300,
    mip_gap=0.01,
    threads=None,
    presolve=True,
    verbose=False,
    load_solution=True,
):
    opt = SolverFactory("gurobi")
    if opt is None or (hasattr(opt, "available") and not opt.available(False)):
        raise RuntimeError("Gurobi solver is not available.")

    n_threads = threads
    if n_threads is None:
        n_threads = max(1, (os.cpu_count() or 2) - 1)

    opt.options["TimeLimit"] = float(time_limit)
    opt.options["MIPGap"] = float(mip_gap)
    opt.options["Threads"] = int(n_threads)
    opt.options["Presolve"] = 2 if presolve else 0

    # Helps Gurobi use current variable values as a MIP start
    results = opt.solve(
        m,
        tee=bool(verbose),
        load_solutions=False,
        warmstart=True,
    )

    loaded = False
    if load_solution:
        loaded = _load_solution_if_available(m, results)

    term = None
    try:
        term = results.solver.termination_condition
    except Exception:
        pass

    best_bound = _extract_best_bound(results, m)

    best_feasible_obj = None
    if loaded:
        try:
            best_feasible_obj = float(value(m.obj))
        except Exception:
            best_feasible_obj = None

    # Adapt to your old interface
    return SimpleNamespace(
        termination_condition=term,
        best_objective_bound=best_bound,
        best_feasible_objective=best_feasible_obj,
        raw_results=results,
    )


def _safe_set_config(opt, attr: str, value_) -> None:
    try:
        setattr(opt.config, attr, value_)
    except Exception:
        pass


def _solve_with_highs_fallback(
    m,
    time_limit=300,
    mip_gap=0.01,
    threads=1,
    presolve=True,
    verbose=False,
    load_solution=True,
):
    if Highs is None:
        raise RuntimeError("HiGHS fallback is not available.")

    opt = Highs()

    n_threads = threads
    if threads is None:
         n_threads = 1


    _safe_set_config(opt, "stream_solver", bool(verbose))
    _safe_set_config(opt, "time_limit", float(time_limit))
    _safe_set_config(opt, "mip_rel_gap", float(mip_gap))
    _safe_set_config(opt, "threads", int(n_threads))
    _safe_set_config(opt, "presolve", "on" if presolve else "off")
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


def solve_model(
    m,
    time_limit=300,
    mip_gap=0.01,
    threads=1,
    presolve=True,
    verbose=False,
    load_solution=True,
    solver_name="gurobi",
):
    """
    Gurobi-first solver wrapper.

    Keeps the old public interface so the rest of the project still works.
    """
    name = str(solver_name).lower().strip()

    if name == "gurobi":
        try:
            return _solve_with_gurobi(
                m,
                time_limit=time_limit,
                mip_gap=mip_gap,
                threads=threads,
                presolve=presolve,
                verbose=verbose,
                load_solution=load_solution,
            )
        except Exception as e:
            print(f"Gurobi failed/unavailable, falling back to HiGHS: {e}")
            return _solve_with_highs_fallback(
                m,
                time_limit=time_limit,
                mip_gap=mip_gap,
                threads=threads,
                presolve=presolve,
                verbose=verbose,
                load_solution=load_solution,
            )

    if name == "highs":
        return _solve_with_highs_fallback(
            m,
            time_limit=time_limit,
            mip_gap=mip_gap,
            threads=threads,
            presolve=presolve,
            verbose=verbose,
            load_solution=load_solution,
        )

    raise ValueError(f"Unknown solver_name: {solver_name}")