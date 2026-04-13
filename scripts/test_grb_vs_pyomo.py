from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC  = ROOT / "src"
DATA_DIR = ROOT / "data" / "input"
sys.path.insert(0, str(SRC))

from evcs.model     import build_multi_period_model
from evcs.solve     import solve_model
from evcs.model_grb import build_multi_period_model_grb
from evcs.solve_grb import solve_model_grb
from evcs.instance  import generate_instance
from evcs.geom      import build_arcs


# --- csv loader ---
def build_instance_from_csv(csv_path: Path, T: int, seed: int, D_km: float) -> dict:
    df = pd.read_csv(csv_path)
    coords_deg = df[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(dtype=float)
    coords_km  = coords_deg.copy()
    coords_km[:, 0] *= 111.0
    coords_km[:, 1] *= 111.0

    pop = df["Aggregated_Population"].to_numpy(dtype=float)
    pop = np.maximum(pop, 0.0)
    if pop.sum() <= 0:
        raise ValueError(f"Population sums to zero: {csv_path}")
    pop_share = pop / pop.sum()
    M = coords_km.shape[0]
    base = pop_share * M

    rng = np.random.default_rng(seed)
    demand_IT = np.zeros((T, M), dtype=float)
    for t in range(T):
        season_factor = 1.0 + 0.2 * np.sin(2 * np.pi * t / T)
        noise = rng.normal(0.0, 0.05, size=M)
        demand_IT[t] = np.maximum(0.0, base * season_factor * (1.0 + noise))

    distIJ, in_range, Ji, Ij = build_arcs(coords_km, coords_km, D=D_km, forbid_self=False)
    return dict(coords_I=coords_km, coords_J=coords_km.copy(),
                demand_IT=demand_IT, distIJ=distIJ,
                in_range=in_range, Ji=Ji, Ij=Ij, M=M, N=M)


# --- run one ---
def _run_one(label, inst, N, T, Q, P_per_period, max_chargers_per_site,
             method_name, allow_multi_charger, cumulative_install,
             mip_gap, time_limit, tol, verbose):
    M_actual  = len(inst["coords_I"])
    demand_IT = np.asarray(inst["demand_IT"], dtype=float)
    P_T       = [P_per_period] * T

    print(f"\n{'='*65}\n  {label}\n{'='*65}")

    params = dict(
        M=M_actual, N=len(inst["coords_J"]), T=T,
        in_range=inst["in_range"], Ji=inst["Ji"], Ij=inst["Ij"],
        demand_IT=demand_IT, Q=Q, P_T=P_T,
        distIJ=inst["distIJ"], method_name=method_name,
        allow_multi_charger=allow_multi_charger,
        max_chargers_per_site=max_chargers_per_site,
        cumulative_install=cumulative_install,
    )

    print("\n[Pyomo]  building …", end="  ", flush=True)
    t0 = time.perf_counter()
    m_pyomo = build_multi_period_model(**params)
    print(f"done ({time.perf_counter()-t0:.2f}s)")

    print("[Pyomo]  solving …", end="  ", flush=True)
    t0 = time.perf_counter()
    res_p = solve_model(m_pyomo, time_limit=time_limit, mip_gap=mip_gap, verbose=verbose)
    pt    = time.perf_counter() - t0
    print(f"done ({pt:.2f}s)  obj={res_p.best_feasible_objective}  term={res_p.termination_condition}")

    print("\n[GRB]    building …", end="  ", flush=True)
    t0 = time.perf_counter()
    m_grb = build_multi_period_model_grb(**params)
    print(f"done ({time.perf_counter()-t0:.2f}s)")

    print("[GRB]    solving …", end="  ", flush=True)
    t0 = time.perf_counter()
    res_g = solve_model_grb(m_grb, time_limit=time_limit, mip_gap=mip_gap, verbose=verbose)
    gt    = time.perf_counter() - t0
    print(f"done ({gt:.2f}s)  obj={res_g.best_feasible_objective}  term={res_g.termination_condition}")

    po, go = res_p.best_feasible_objective, res_g.best_feasible_objective
    assert po is not None, "Pyomo: no feasible solution"
    assert go is not None, "GRB: no feasible solution"

    diff     = abs(po - go)
    rel_diff = diff / max(abs(po), abs(go), 1.0)
    print(f"\n  Pyomo: {po:.8f}  GRB: {go:.8f}  abs={diff:.2e}  rel={rel_diff:.2e}  tol={tol:.0e}")
    assert rel_diff <= tol, f"Objective mismatch: Pyomo={po}  GRB={go}  rel={rel_diff:.2e}"

    x_mis = 0
    for j in range(len(inst["coords_J"])):
        for t in range(T):
            try: px = round(float(m_pyomo.x[j, t].value or 0))
            except Exception: px = 0
            try: gx = round(float(m_grb._meta.x[j, t].X if m_grb._gm.SolCount > 0 else 0))
            except Exception: gx = 0
            if px != gx:
                x_mis += 1
                if x_mis <= 5:
                    print(f"  x[{j},{t}]: Pyomo={px}  GRB={gx}")

    print(f"  {'All x[j,t] match' if not x_mis else f'{x_mis} x[j,t] differ (alt optima)'}")
    print(f"\n  PASS  [{label}]")
    return dict(tag=label, pyomo_obj=po, grb_obj=go, diff=diff, rel_diff=rel_diff,
                x_mismatches=x_mis, pyomo_time=pt, grb_time=gt)


# --- suites ---
QUICK_CASES = [
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
    results, failures = [], []
    for case in cases:
        M, N, T, Q, D, seed, P, mc, method, multi, cumul = case
        try:
            inst = generate_instance(N_sites=N, T=T, seed=seed)
            distIJ, in_range, Ji, Ij = build_arcs(inst["coords_I"], inst["coords_J"], D=D)
            inst.update(distIJ=distIJ, in_range=in_range, Ji=Ji, Ij=Ij)
            label = f"synthetic M={len(inst['coords_I'])} N={N} T={T} Q={Q} D={D} seed={seed} method={method}"
            r = _run_one(label, inst, N, T, Q, P, mc, method, multi, cumul,
                         mip_gap, time_limit, tol=1e-4, verbose=verbose)
            results.append(r)
        except AssertionError as e:
            print(f"\n  FAIL: {e}")
            failures.append((str(case), str(e)))
        except Exception as e:
            import traceback; traceback.print_exc()
            failures.append((str(case), str(e)))
    _print_summary(results, failures)


# --- tsv loader ---
def load_config_tsv(tsv_path: Path) -> list[dict]:
    rows = []
    with open(tsv_path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            rows.append(dict(row_id=parts[0].strip(), csv_file=parts[1].strip(),
                             seed=int(parts[2]), D_km=float(parts[3]),
                             T=int(parts[4]), Q=float(parts[5])))
    return rows


def run_config_suite(tsv_path, rows_filter=None, P_per_period=8,
                     max_chargers_per_site=6, method_name="none",
                     allow_multi_charger=True, cumulative_install=True,
                     mip_gap=0.0, time_limit=300, tol=1e-4, verbose=False):
    config_rows = load_config_tsv(tsv_path)
    if not config_rows:
        raise ValueError(f"No valid rows in {tsv_path}")
    if rows_filter:
        config_rows = [r for r in config_rows if r["row_id"] in rows_filter]
        if not config_rows:
            raise ValueError(f"None of {rows_filter} found in {tsv_path}")

    print(f"\n{len(config_rows)} instance(s) from {tsv_path.name}  "
          f"P={P_per_period}  max_c={max_chargers_per_site}  method={method_name}")

    results, failures = [], []
    for row in config_rows:
        csv_path = DATA_DIR / row["csv_file"]
        if not csv_path.exists():
            msg = f"CSV not found: {csv_path}"
            print(f"\n  SKIP row {row['row_id']}: {msg}")
            failures.append((row["row_id"], msg))
            continue
        label = f"row={row['row_id']}  {row['csv_file']}  seed={row['seed']}  D={row['D_km']}  T={row['T']}  Q={row['Q']}  method={method_name}"
        try:
            print(f"\nLoading {row['csv_file']} …", end="  ", flush=True)
            inst = build_instance_from_csv(csv_path, T=row["T"], seed=row["seed"], D_km=row["D_km"])
            print(f"M={inst['M']}  |A|={len(inst['in_range'])}")
            r = _run_one(label, inst, inst["N"], row["T"], row["Q"],
                         P_per_period, max_chargers_per_site,
                         method_name, allow_multi_charger, cumulative_install,
                         mip_gap, time_limit, tol, verbose)
            results.append(r)
        except AssertionError as e:
            print(f"\n  FAIL: {e}")
            failures.append((row["row_id"], str(e)))
        except Exception as e:
            import traceback; traceback.print_exc()
            failures.append((row["row_id"], str(e)))

    _print_summary(results, failures)


# --- summary ---
def _print_summary(results, failures):
    print(f"\n{'='*65}")
    print(f"  {len(results)} passed  {len(failures)} failed")
    for r in results:
        s = "+" if r["rel_diff"] <= 1e-4 else "~"
        print(f"  {s} {r['tag'][:52]:<52}  d={r['rel_diff']:.1e}  "
              f"Pyomo {r['pyomo_time']:.1f}s / GRB {r['grb_time']:.1f}s")
    if failures:
        print("\n  FAILURES:")
        for key, msg in failures:
            print(f"    [{key}] {msg}")
    print(f"{'='*65}\n")
    if failures:
        raise SystemExit(f"{len(failures)} test(s) failed")
    print("All tests passed.")


# --- cli ---
def _parse():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--config", type=Path, metavar="TSV")
    mode.add_argument("--full",   action="store_true")
    p.add_argument("--rows",        type=str,   default=None)
    p.add_argument("--P",           type=int,   default=8)
    p.add_argument("--max-chargers",type=int,   default=6, dest="max_chargers")
    p.add_argument("--method",      type=str,   default="none",
                   choices=["none","closest_only","closest_priority","system_optimum"])
    p.add_argument("--mip-gap",     type=float, default=0.0,  dest="mip_gap")
    p.add_argument("--time-limit",  type=float, default=300,  dest="time_limit")
    p.add_argument("--verbose",     action="store_true")
    p.add_argument("--M",    type=int,   default=25)
    p.add_argument("--N",    type=int,   default=25)
    p.add_argument("--T",    type=int,   default=3)
    p.add_argument("--Q",    type=float, default=10.0)
    p.add_argument("--D",    type=float, default=0.35)
    p.add_argument("--seed", type=int,   default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if args.config:
        rows_filter = [r.strip() for r in args.rows.split(",")] if args.rows else None
        run_config_suite(tsv_path=args.config, rows_filter=rows_filter,
                         P_per_period=args.P, max_chargers_per_site=args.max_chargers,
                         method_name=args.method, mip_gap=args.mip_gap,
                         time_limit=args.time_limit, verbose=args.verbose)

    elif args.full:
        print("FULL suite …")
        run_suite(FULL_CASES, mip_gap=args.mip_gap, time_limit=args.time_limit, verbose=args.verbose)

    else:
        defaults = dict(M=25, N=25, T=3, Q=10.0, D=0.35, seed=42)
        is_custom = any(getattr(args, k) != defaults[k] for k in defaults)

        if is_custom:
            inst = generate_instance(N_sites=args.N, T=args.T, seed=args.seed)
            distIJ, in_range, Ji, Ij = build_arcs(inst["coords_I"], inst["coords_J"], D=args.D)
            inst.update(distIJ=distIJ, in_range=in_range, Ji=Ji, Ij=Ij)
            label = f"synthetic M={len(inst['coords_I'])} N={args.N} T={args.T} Q={args.Q} D={args.D} seed={args.seed} method={args.method}"
            _run_one(label, inst, args.N, args.T, args.Q,
                     args.P, args.max_chargers, args.method, True, True,
                     args.mip_gap, args.time_limit, tol=1e-4, verbose=args.verbose)
        else:
            print("QUICK suite (--full for more) …")
            run_suite(QUICK_CASES, mip_gap=args.mip_gap, time_limit=args.time_limit, verbose=args.verbose)
