"""
extend_multiseed_cache_effect.py
=================================
Extends the cache-effect (heap_cache - heap_only) seed count for Roma,
Napoli, and Genova from n=3 seeds to n=7, matching the seed depth already
collected for Milano and Torino (11/22/33 + 44/55/66/77) so that the
arc-density-vs-cache-effect predictor (r=0.807 on seed-averaged deltas,
see project_gini_cache_effect_hypothesis memory) can be re-tested for
seed-to-seed robustness across ALL 5 non-ceiling cities at equal seed depth.

Reuses the exact code path already validated for each city (same P_t,
same configs, same budget per city) rather than re-deriving anything:
  - Roma, Napoli: controlled_reconstruction_experiment.py path, budget=120s
    (matches controlled_per_run.csv seeds 11/22/33)
  - Genova: budget=60s (matches gini_cache_effect_per_run_genova60.csv;
    120s hits a ceiling effect for this instance size, see memory)

Only NEW seeds (44/55/66/77) are run here; existing seeds are left in their
original CSVs untouched. Output is a separate tagged CSV so nothing gets
clobbered:
  results/diagnostics/gini_cache_effect/multiseed_extra_per_run.csv

Usage:
    python scripts/extend_multiseed_cache_effect.py
    python scripts/extend_multiseed_cache_effect.py --seeds 44 55 --budget-override 5   # smoke test
"""
import sys, argparse
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(Path(__file__).resolve().parent))

import controlled_reconstruction_experiment as cre

OUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "gini_cache_effect"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (city, csv_name, budget_s) — budgets match what's already on disk for each city
CITY_SPECS = [
    ("Roma",   cre.INSTANCES["Roma"],                           120.0),
    ("Napoli", cre.INSTANCES["Napoli"],                          120.0),
    ("Genova", "center_276_Genova_k825.csv",                     60.0),
]

CONFIGS = [c for c in cre.CONFIGS if c[0] in ("heap_only", "heap_cache")]

DEFAULT_SEEDS = [44, 55, 66, 77]
NREF = 3525


OUT_CSV = OUT_DIR / "multiseed_extra_per_run.csv"


def save_incremental(rows):
    """Append rows to OUT_CSV immediately, deduping on (city,label,seed), so a
    crash later in the run (e.g. process killed after many iterations — this
    script previously lost an entire completed Roma+partial-Napoli batch to
    exactly that) doesn't lose already-computed results."""
    df = pd.DataFrame(rows)
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(
            subset=["city", "label", "seed"], keep="last")
    df.to_csv(OUT_CSV, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", default=[c for c, _, _ in CITY_SPECS],
                     choices=[c for c, _, _ in CITY_SPECS])
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--budget-override", type=float, default=None,
                     help="override all per-city budgets (for smoke-testing only)")
    args = ap.parse_args()

    P_t = max(4, NREF // 30)
    P_T = [P_t] * cre.T_PERIODS
    print(f"N_ref={NREF} -> P_t={P_t} (matches existing runs)  new seeds={args.seeds}  cities={args.city}")

    specs = [(c, csv, b) for c, csv, b in CITY_SPECS if c in args.city]
    for city, csv_name, budget in specs:
        budget = args.budget_override if args.budget_override is not None else budget
        print(f"\n{'='*70}\n  {city}   budget={budget}s/config/seed   seeds={args.seeds}\n{'='*70}")
        for seed in args.seeds:
            print(f"  Loading instance (seed={seed})...", end=" ", flush=True)
            inst, Ij_int, ue_eval, ue_fn, M = cre.load_instance(csv_name, seed, P_T)
            print(f"done  N={M}")
            seed_rows = []
            for label, R, use_heap, use_dw in CONFIGS:
                r = cre.run_config(label, R, use_heap, use_dw, inst, Ij_int, P_T,
                                    ue_fn, budget, seed)
                r["city"] = city
                r["N"] = M
                seed_rows.append(r)
                print(f"    [seed={seed}] {label:<11} iters={r['iters']:>4}  "
                      f"best={r['best_score']:>9.3f}")
            ue_eval.dispose()
            save_incremental(seed_rows)
            print(f"    -> saved (city={city}, seed={seed}) to {OUT_CSV}")

    print("\nDone.")


if __name__ == "__main__":
    main()
