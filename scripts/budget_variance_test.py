"""
budget_variance_test.py
========================
Cheap pre-check before investing in a full budget-slicing experiment: does a
longer per-seed reconstruction budget shrink the within-city seed-to-seed
variance of the heap_cache-minus-heap_only delta, or is the variance itself
a property of the effect (in which case more time per seed won't help)?

See project_arc_density_seed_robustness_refutation memory: at the current
120s budget, every one of 5 cities showed delta sign-flips across 7 seeds,
with within-city std (21-51) comparable to or exceeding the entire
across-city spread in means. Before re-running the whole 5-city x N-seed
grid at a longer budget (expensive), test ONE city at a longer budget and
compare its variance to the existing 120s baseline for the same 7 seeds.

City chosen: Napoli — it had the highest coefficient of variation (17.3) of
the 5 non-ceiling cities, i.e. the "least stable" signal at 120s, so it's
the most informative single case to re-test.

Writes to a SEPARATE csv (not multiseed_extra_per_run.csv) so the 120s
baseline data is never touched/clobbered, and includes a budget_s column so
multiple budget levels can coexist in one file.

Usage:
    python scripts/budget_variance_test.py --city Napoli --seeds 11 --budget 300
    (looped externally, one seed per subprocess call, to avoid the memory
    buildup that crashed the first multiseed-extension run — see memory)
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
OUT_CSV = OUT_DIR / "budget_variance_per_run.csv"

CONFIGS = [c for c in cre.CONFIGS if c[0] in ("heap_only", "heap_cache")]
NREF = 3525

CITY_CSV = {
    "Napoli": cre.INSTANCES["Napoli"],
    "Roma":   cre.INSTANCES["Roma"],
    "Milano": cre.INSTANCES["Milano"],
    "Torino": cre.INSTANCES["Torino"],
    "Genova": "center_276_Genova_k825.csv",
}


def save_incremental(rows):
    df = pd.DataFrame(rows)
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(
            subset=["city", "label", "seed", "budget_s"], keep="last")
    df.to_csv(OUT_CSV, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="Napoli", choices=list(CITY_CSV.keys()))
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--budget", type=float, required=True)
    args = ap.parse_args()

    P_t = max(4, NREF // 30)
    P_T = [P_t] * cre.T_PERIODS
    csv_name = CITY_CSV[args.city]

    print(f"{args.city}  budget={args.budget}s/config/seed  seeds={args.seeds}  P_t={P_t}")
    for seed in args.seeds:
        print(f"  Loading instance (seed={seed})...", end=" ", flush=True)
        inst, Ij_int, ue_eval, ue_fn, M = cre.load_instance(csv_name, seed, P_T)
        print(f"done  N={M}")
        seed_rows = []
        for label, R, use_heap, use_dw in CONFIGS:
            r = cre.run_config(label, R, use_heap, use_dw, inst, Ij_int, P_T,
                                ue_fn, args.budget, seed)
            r["city"] = args.city
            r["N"] = M
            r["budget_s"] = args.budget
            seed_rows.append(r)
            print(f"    [seed={seed}] {label:<11} iters={r['iters']:>4}  "
                  f"best={r['best_score']:>9.3f}")
        ue_eval.dispose()
        save_incremental(seed_rows)
        print(f"    -> saved (city={args.city}, seed={seed}, budget={args.budget}) to {OUT_CSV}")

    print("\nDone.")


if __name__ == "__main__":
    main()
