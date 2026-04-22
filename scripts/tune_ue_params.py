# tune_ue_params.py  --  Vicenza N=125
#
# Distances are in km (coords x 111, matching build_instance_from_csv).
# mu = Q + eps = 20.0001 (matching Q=20 from collect_trajectories.py).
# Sweep: noopt_cost (km) x alpha_wait
# Acceptance criterion: greedy served_frac > 0.50.

import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from evcs.geom import build_arcs
from evcs.ue_evaluator import UEEvaluator


# ── instance ──────────────────────────────────────────────────────────────────

D_KM   = 2.0    # range cutoff in km (matches collect_trajectories.py default)
Q      = 20.0   # capacity per server (matches collect_trajectories.py Q=20)
MU     = Q + 1e-4
S_MAX  = 4
BUDGET = 15

df       = pd.read_csv("data/input/center_102_Vicenza_k125.csv")
coords_d = df[["Centroid_Longitude", "Centroid_Latitude"]].values
coords   = coords_d * 111.0          # degrees -> km (approx, flat-earth)

d_raw  = df["Aggregated_Population"].values
pop    = np.maximum(d_raw, 0.0)
N      = len(pop)
d      = (pop / pop.sum()) * N       # normalized so sum = N (= 125), mean = 1

distIJ, in_range, Ji, Ij = build_arcs(coords, coords, D=D_KM, forbid_self=False)

total_d = float(np.sum(d))

print(f"instance   Vicenza N={N}  D={D_KM} km  Q={Q}  budget={BUDGET}  s_max={S_MAX}")
print(f"distances  min={distIJ[distIJ>0].min():.2f} km  "
      f"max={distIJ.max():.2f} km  "
      f"median={np.median(distIJ[distIJ>0]):.2f} km")
print(f"in_range pairs: {len(in_range)}  (D={D_KM} km cutoff)")
print(f"total demand: {total_d:.1f}  |  max capacity: {BUDGET * MU:.1f}")
print()


# ── placements ────────────────────────────────────────────────────────────────

def greedy_u():
    servers, remaining = np.zeros(N, dtype=int), BUDGET
    for j in np.argsort(d)[::-1]:
        if remaining <= 0:
            break
        servers[j] = min(S_MAX, remaining)
        remaining -= servers[j]
    return {(j, 0): int(servers[j]) for j in range(N) if servers[j] > 0}


rng = np.random.default_rng(42)

def rand_u():
    servers, remaining = np.zeros(N, dtype=int), BUDGET
    for j in rng.permutation(N):
        if remaining <= 0:
            break
        s = int(rng.integers(1, min(S_MAX, remaining) + 1))
        servers[j] = s
        remaining -= s
    return {(j, 0): int(servers[j]) for j in range(N) if servers[j] > 0}


U_greedy     = greedy_u()
U_rand_batch = [rand_u() for _ in range(4)]
placements   = [U_greedy] + U_rand_batch


# ── sweep ─────────────────────────────────────────────────────────────────────

NOOPT_COSTS = [2.5, 3.0, 4.0]   # km: cost of using no-option (= effective range threshold)
ALPHA_WAITS = [0.1, 0.5, 1.0]   # weight on Erlang-C waiting term

T = 1

print(f"{'noopt_km':>9}  {'alpha_w':>7}  {'greedy%':>8}  "
      f"{'rand_mean%':>10}  {'rand_min%':>10}  verdict")
print("-" * 64)

best_row  = None
best_frac = -1.0

for noopt in NOOPT_COSTS:
    for alpha in ALPHA_WAITS:

        ev = UEEvaluator(
            N=N,
            d=d,
            tau=distIJ,
            mu=MU,
            s_max=S_MAX,
            noopt_cost=noopt,
            alpha_wait=alpha,
            N_bp=50,
        )

        fracs = []
        for U in placements:
            lam, _ = ev.evaluate(U, T=T, cumulative_install=True)
            fracs.append(lam / total_d)

        g_frac    = fracs[0]
        r_fracs   = fracs[1:]
        verdict   = "OK" if g_frac > 0.50 else "low"

        print(f"{noopt:>9.1f}  {alpha:>7.1f}  {g_frac:>8.1%}  "
              f"{np.mean(r_fracs):>10.1%}  {np.min(r_fracs):>10.1%}  {verdict}")

        if g_frac > best_frac:
            best_frac = g_frac
            best_row  = (noopt, alpha)

print()
print(f"Best: noopt_cost={best_row[0]} km  alpha_wait={best_row[1]}"
      f"  greedy served={best_frac:.1%}")
