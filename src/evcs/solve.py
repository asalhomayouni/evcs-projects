# src/evcs/solve.py
import os
from pyomo.contrib.appsi.solvers import Highs
from pyomo.opt import TerminationCondition

def solve_model(
    m,
    time_limit=300,     # seconds (default 5 minutes)
    mip_gap=0.01,       # relative MIP gap (1%)
    threads=None,       # default: all cores - 1
    presolve=True,
    verbose=False
):
    """
    Solve a Pyomo model using the HiGHS solver (Appsi interface).
    Configured with safe defaults for performance and reproducibility.

    Parameters
    ----------
    m : ConcreteModel
        Pyomo model to solve.
    time_limit : float, optional
        Maximum runtime in seconds (default: 300).
    mip_gap : float, optional
        Relative MIP optimality gap (default: 0.01 = 1%).
    threads : int, optional
        Number of threads to use (default: os.cpu_count() - 1).
    presolve : bool, optional
        Whether to enable presolve (default: True).
    verbose : bool, optional
        Whether to print solver logs (default: False).

    Returns
    -------
    res : Results object
        The solver results (contains termination condition, status, etc.).
    """

    opt = Highs()

    # Configure solver options safely
    try:
        opt.config.stream_solver = bool(verbose)
    except Exception:
        pass

    if threads is None:
        threads = max(1, (os.cpu_count() or 2) - 1)

    def safe_set(attr, value):
        try:
            setattr(opt.config, attr, value)
        except Exception:
            pass

    # Apply solver options
    safe_set("time_limit", float(time_limit))
    safe_set("mip_rel_gap", float(mip_gap))
    safe_set("threads", int(threads))
    safe_set("presolve", "on" if presolve else "off")

    # Solve model
    res = opt.solve(m)

    # Report short summary
    term = getattr(res, "termination_condition", "unknown")
    if verbose:
        print(f"✅ Solver finished with status: {term}")

    return res
