def solve_model(m, verbose=False):
    """
    Prefer HiGHS (pure Python wheel) then fall back to GLPK / CBC if installed.
    """
    # 1) HiGHS via appsi
    try:
        from pyomo.contrib.appsi.solvers.highs import Highs
        opt = Highs()
        try: opt.config.stream_solver = bool(verbose)
        except Exception: pass
        return opt.solve(m)
    except Exception:
        pass

    # 2) Classic Pyomo solvers
    from pyomo.environ import SolverFactory
    for name in ("glpk", "cbc"):
        try:
            opt = SolverFactory(name)
            return opt.solve(m, tee=verbose)
        except Exception:
            continue

    raise RuntimeError("No solver available (install highspy OR glpk-utils OR coinor-cbc).")
