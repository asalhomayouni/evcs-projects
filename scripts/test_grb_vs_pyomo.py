"""
test_grb_vs_pyomo.py
====================
Run the Pyomo + Gurobi solver and the native gurobipy solver on the same
instance, then assert that their objective values are identical (within
floating-point tolerance).

Usage
-----
    python scripts/test_grb_vs_pyomo.py                 # small default case
    python scripts/test_grb_vs_pyomo.py --full          # all policy variants
    python scripts/test_grb_vs_pyomo.py --M 50 --N 50   # custom size

NOTE
----
Run this script BEFORE commenting out Pyomo in model.py / solve.py.
Once you are satisfied with the results, proceed with the migration.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ paths
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

# ------------------------------------------------------------------ Pyomo path  (pre-migration)
from evcs.model import build_multi_period_model          # noqa: E402
from evcs.solve import solve_model                       # noqa: E402

# ------------------------------------------------------------------ GRB native path
from evcs.model_grb import build_multi_period_model_grb  # noqa: E402
from evcs.solve_grb import solve_model_grb               # noqa: E402

# ------------------------------------------------------------------ instance helpers
from evcs.instance import generate_instance              # noqa: E402
from evcs.geom import build_arcs                         # noqa: E402


# ===========================================================================
# Core comparison function
# ===========================================================================
def run_comparison(
    M: int = 25,
    N: int = 25,
    T: int = 3,
    Q: float = 10.0,
    D: float = 0.35,
    seed: int = 42,
    P_per_period: int = 5,
    max_chargers_per_site: int = 4,
    method_name: str = "none",
    allow_multi_charger: bool = True,
    cumulative_install: bool = True,
    mip_gap: float = 0.0,
    time_limit: float = 300,
    tol: float = 1e-4,
    verbose: bool = False,
) -> dict:
    """
    Build and solve the same instance with Pyomo+Gurobi and with native GRB.
    Returns a result dict; raises AssertionError on mismatch.
    """
    tag = (f"M={M} N={N} T={T} Q={Q} D={D} seed={seed} "
           f"method={method_name} mc={allow_multi_charger}")
    print(f"\n{'='*65}")
    print(f"  {tag}")
    print(f"{'='*65}")

    # --- build instance ---
    inst = generate_instance(N_sites=N, T=T, seed=seed)
    M_actual = len(inst["coords_I"])
    distIJ, in_range, Ji, Ij = build_arcs(
        inst["coords_I"], inst["coords_J"], D=D
    )
    demand_IT = np.asarray(inst["demand_IT"], dtype=float)  # (T, M)
    P_T = [P_per_period] * T

    shared_params = dict(
        M=M_actual, N=N, T=T,
        in_range=in_range, Ji=Ji, Ij=Ij,
        demand_IT=demand_IT,
        Q=Q, P_T=P_T,
        distIJ=distIJ,
        method_name=method_name,
        allow_multi_charger=allow_multi_charger,
        max_chargers_per_site=max_chargers_per_site,
        cumulative_install=cumulative_install,
    )

    # ----------------------------------------------------------------
    # PYOMO build + solve
    # ----------------------------------------------------------------
    print("\n[Pyomo]  building model …", end="  ", flush=True)
    t0 = time.perf_counter()
    m_pyomo = build_multi_period_model(**shared_params)
    print(f"done  ({time.perf_counter()-t0:.2f}s)")

    print("[Pyomo]  solving …", end="  ", flush=True)
    t0 = time.perf_counter()
    res_pyomo = solve_model(
        m_pyomo,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
    )
    pyomo_time = time.perf_counter() - t0
    pyomo_obj = res_pyomo.best_feasible_objective
    pyomo_bound = res_pyomo.best_objective_bound
    pyomo_term = res_pyomo.termination_condition
    print(f"done  ({pyomo_time:.2f}s)  obj={pyomo_obj}  "
          f"bound={pyomo_bound}  term={pyomo_term}")

    # ----------------------------------------------------------------
    # GRB native build + solve
    # ----------------------------------------------------------------
    print("\n[GRB]    building model …", end="  ", flush=True)
    t0 = time.perf_counter()
    m_grb = build_multi_period_model_grb(**shared_params)
    print(f"done  ({time.perf_counter()-t0:.2f}s)")

    print("[GRB]    solving …", end="  ", flush=True)
    t0 = time.perf_counter()
    res_grb = solve_model_grb(
        m_grb,
        time_limit=time_limit,
        mip_gap=mip_gap,
        verbose=verbose,
    )
    grb_time = time.perf_counter() - t0
    grb_obj = res_grb.best_feasible_objective
    grb_bound = res_grb.best_objective_bound
    grb_term = res_grb.termination_condition
    print(f"done  ({grb_time:.2f}s)  obj={grb_obj}  "
          f"bound={grb_bound}  term={grb_term}")

    # ----------------------------------------------------------------
    # Assert both found a solution
    # ----------------------------------------------------------------
    assert pyomo_obj is not None, \
        f"[Pyomo] no feasible solution  (term={pyomo_term})"
    assert grb_obj is not None, \
        f"[GRB] no feasible solution  (term={grb_term})"

    # ----------------------------------------------------------------
    # Objective comparison
    # ----------------------------------------------------------------
    diff = abs(pyomo_obj - grb_obj)
    scale = max(abs(pyomo_obj), abs(grb_obj), 1.0)
    rel_diff = diff / scale

    print(f"\n  Pyomo obj  : {pyomo_obj:.8f}")
    print(f"  GRB   obj  : {grb_obj:.8f}")
    print(f"  abs diff   : {diff:.2e}   rel diff: {rel_diff:.2e}  "
          f"(tol={tol:.0e})")

    assert rel_diff <= tol, (
        f"Objective mismatch!\n"
        f"  Pyomo = {pyomo_obj}\n"
        f"  GRB   = {grb_obj}\n"
        f"  diff  = {diff}  (rel {rel_diff:.2e} > tol {tol:.0e})"
    )

    # ----------------------------------------------------------------
    # x[j,t] solution comparison
    # ----------------------------------------------------------------
    meta = m_grb._meta
    x_mismatches = 0
    for j in range(N):
        for t in range(T):
            try:
                pyomo_x = round(float(m_pyomo.x[j, t].value or 0))
            except Exception:
                pyomo_x = 0
            try:
                grb_x_raw = meta.x[j, t].X if m_grb._gm.SolCount > 0 else 0
                grb_x = round(float(grb_x_raw or 0))
            except Exception:
                grb_x = 0
            if pyomo_x != grb_x:
                x_mismatches += 1
                if x_mismatches <= 8:
                    print(f"  x[{j},{t}]: Pyomo={pyomo_x}  GRB={grb_x}")

    if x_mismatches:
        print(f"  NOTE: {x_mismatches} x[j,t] values differ "
              f"(alternative optima — objective still matches)")
    else:
        print("  All x[j,t] values identical ✓")

    # ----------------------------------------------------------------
    # u[j,t] comparison (multi-charger only)
    # ----------------------------------------------------------------
    if allow_multi_charger:
        u_mismatches = 0
        for j in range(N):
            for t in range(T):
                try:
                    pyomo_u = round(float(m_pyomo.u[j, t].value or 0))
                except Exception:
                    pyomo_u = 0
                try:
                    grb_u_raw = meta.u[j, t].X if m_grb._gm.SolCount > 0 else 0
                    grb_u = round(float(grb_u_raw or 0))
                except Exception:
                    grb_u = 0
                if pyomo_u != grb_u:
                    u_mismatches += 1
        if u_mismatches:
            print(f"  NOTE: {u_mismatches} u[j,t] values differ "
                  f"(alternative optima)")
        else:
            print("  All u[j,t] values identical ✓")

    print(f"\n  ✓ PASS  [{tag}]")

    return dict(
        tag=tag,
        pyomo_obj=pyomo_obj,
        grb_obj=grb_obj,
        diff=diff,
        rel_diff=rel_diff,
        x_mismatches=x_mismatches,
        pyomo_time=pyomo_time,
        grb_time=grb_time,
    )


# ===========================================================================
# Test suite
# ===========================================================================
QUICK_CASES = [
    # (M, N, T, Q, D, seed, P_per_period, max_c, method, multi_c, cumul)
    (25, 25, 3, 10.0, 0.35, 42,  5, 4, "none",             True,  True),
    (25, 25, 3, 10.0, 0.35, 42,  5, 4, "closest_priority", True,  True),
    (25, 25, 3, 10.0, 0.35, 42,  5, 4, "system_optimum",   True,  True),
    (25, 25, 3, 10.0, 0.35, 42,  5, 4, "closest_only",     True,  True),
    (20, 20, 2, 15.0, 0.40, 7,   4, 3, "none",             False, False),
]

FULL_CASES = QUICK_CASES + [
    (40, 40, 4, 10.0, 0.30, 99,  6, 5, "none",             True,  True),
    (40, 40, 4, 10.0, 0.30, 99,  6, 5, "closest_priority", True,  True),
    (40, 40, 4, 10.0, 0.30, 99,  6, 5, "system_optimum",   True,  True),
    (30, 30, 3, 20.0, 0.45, 13,  8, 6, "none",             True,  False),
]


def run_suite(cases, mip_gap=0.0, time_limit=300, verbose=False):
    results = []
    failures = []
    for case in cases:
        M, N, T, Q, D, seed, P, mc, method, multi, cumul = case
        try:
            r = run_comparison(
                M=M, N=N, T=T, Q=Q, D=D, seed=seed,
                P_per_period=P,
                max_chargers_per_site=mc,
                method_name=method,
                allow_multi_charger=multi,
                cumulative_install=cumul,
                mip_gap=mip_gap,
                time_limit=time_limit,
                verbose=verbose,
            )
            results.append(r)
        except AssertionError as e:
            print(f"\n  ✗ FAIL: {e}")
            failures.append((case, str(e)))
        except Exception as e:
            print(f"\n  ✗ ERROR: {e}")
            failures.append((case, str(e)))

    # summary
    print(f"\n{'='*65}")
    print(f"  Summary: {len(results)} passed, {len(failures)} failed")
    if results:
        for r in results:
            status = "✓" if r["rel_diff"] <= 1e-4 else "~"
            print(f"  {status} {r['tag'][:55]:<55} "
                  f"Δ={r['rel_diff']:.1e}  "
                  f"Pyomo {r['pyomo_time']:.1f}s / GRB {r['grb_time']:.1f}s")
    if failures:
        print("\n  FAILURES:")
        for case, msg in failures:
            print(f"    {case}")
            print(f"      {msg}")
    print(f"{'='*65}\n")

    if failures:
        raise SystemExit(f"{len(failures)} test(s) failed")
    print("All tests PASSED.  Safe to comment out Pyomo in model.py / solve.py.")


# ===========================================================================
# Entry point
# ===========================================================================
def _parse():
    p = argparse.ArgumentParser(
        description="Compare Pyomo+Gurobi vs native gurobipy on EVCS instances."
    )
    p.add_argument("--full",    action="store_true",
                   help="run extended test suite (more instances)")
    p.add_argument("--M",       type=int,   default=25)
    p.add_argument("--N",       type=int,   default=25)
    p.add_argument("--T",       type=int,   default=3)
    p.add_argument("--Q",       type=float, default=10.0)
    p.add_argument("--D",       type=float, default=0.35)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--P",       type=int,   default=5,
                   help="chargers budget per period")
    p.add_argument("--method",  type=str,   default="none",
                   choices=["none", "closest_only", "closest_priority",
                             "system_optimum"])
    p.add_argument("--mip-gap", type=float, default=0.0)
    p.add_argument("--time-limit", type=float, default=300)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.full:
        print("Running FULL test suite …")
        run_suite(FULL_CASES, mip_gap=args.mip_gap,
                  time_limit=args.time_limit, verbose=args.verbose)
    else:
        # --M / --N / --seed etc. override → single custom run;
        # otherwise fall through to the quick suite.
        custom = dict(
            M=args.M, N=args.N, T=args.T, Q=args.Q, D=args.D,
            seed=args.seed, P_per_period=args.P,
            method_name=args.method,
            mip_gap=args.mip_gap, time_limit=args.time_limit,
            verbose=args.verbose,
        )
        defaults = dict(M=25, N=25, T=3, Q=10.0, D=0.35, seed=42,
                        P_per_period=5, method_name="none")
        is_custom = any(custom[k] != defaults[k] for k in defaults)

        if is_custom:
            run_comparison(**custom)
        else:
            print("Running QUICK test suite (use --full for more cases) …")
            run_suite(QUICK_CASES, mip_gap=args.mip_gap,
                      time_limit=args.time_limit, verbose=args.verbose)
