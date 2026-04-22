# compare_ue_grb.py  —  Vicenza N=125

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import spearmanr
import pandas as pd

sys.path.insert(0, "src")
from evcs.geom import build_arcs
from evcs.full_eval_grb import GRBEvaluator
from evcs.ue_evaluator import UEEvaluator

try:
    import gurobipy as gp
    from gurobipy import GRB
    from evcs.model_grb import _get_env
except ImportError:
    pass


# params  (distances in km, matching collect_trajectories.py / build_instance_from_csv)

D_KM        = 2.0    # range cutoff km
Q           = 20.0   # capacity per server for GRB (hard capacity)
MU          = 7.5    # Erlang-C service rate per server (rho~0.78 at greedy load)
S_MAX       = 4
BUDGET      = 15
N_RAND      = 5
NOOPT_COST  = 2.0    # km: matches D_KM so UE and GRB have the same effective range
ALPHA_WAIT  = 1.0    # full weight on waiting cost so queueing effects are visible


# instance

df        = pd.read_csv("data/input/center_102_Vicenza_k125.csv")
coords_d  = df[["Centroid_Longitude", "Centroid_Latitude"]].values
coords    = coords_d * 111.0          # degrees -> km (flat-earth approx)
d_raw     = df["Aggregated_Population"].values
pop       = np.maximum(d_raw, 0.0)
N         = len(pop)
d         = (pop / pop.sum()) * N     # sum = N, mean = 1
T         = 1
demand_TM = d.reshape(1, N)

distIJ, in_range, Ji, Ij = build_arcs(coords, coords, D=D_KM, forbid_self=False)


# evaluators

grb_ev = GRBEvaluator(M=N, N=N, T=T, arcs=in_range, demand_TM=demand_TM,
                      Q=Q, cumulative_install=True)
ue_ev  = UEEvaluator(N=N, d=d, tau=distIJ, mu=MU, s_max=S_MAX,
                     noopt_cost=NOOPT_COST, alpha_wait=ALPHA_WAIT, N_bp=50)


# helpers

def make_u(rng, budget=BUDGET, s_max=S_MAX):
    servers, remaining = np.zeros(N, dtype=int), budget
    for j in rng.permutation(N):
        if remaining <= 0: break
        s = int(rng.integers(1, min(s_max, remaining) + 1))
        servers[j] = s
        remaining -= s
    return {(j, 0): int(servers[j]) for j in range(N) if servers[j] > 0}

def greedy_u(budget=BUDGET, s_max=S_MAX):
    servers, remaining = np.zeros(N, dtype=int), budget
    for j in np.argsort(d)[::-1]:
        if remaining <= 0: break
        servers[j] = min(s_max, remaining)
        remaining -= servers[j]
    return {(j, 0): int(servers[j]) for j in range(N) if servers[j] > 0}

def solve_ue_full(U_dict):
    """Re-solve UE LP and return (lam_j, y_noopt_i, routed_frac_i)."""
    s_j    = ue_ev._extract_servers(U_dict, T, True)
    open_j = [j for j in range(N) if s_j[j] > 0]
    if not open_j:
        return np.zeros(N), np.array(d.copy()), np.zeros(N)

    gm      = gp.Model(env=_get_env()); gm.setParam("OutputFlag", 0)
    y       = gm.addVars(N, N, lb=0.0, name="y")
    y_noopt = gm.addVars(N, lb=0.0, name="yn")
    lam     = gm.addVars(N, lb=0.0, name="lam")
    phi     = gm.addVars(N, lb=0.0, name="phi")

    for j in range(N):
        if s_j[j] == 0:
            for i in range(N): y[i, j].UB = 0.0
            lam[j].UB = 0.0; phi[j].UB = 0.0
    gm.update()

    gm.setObjective(
        gp.quicksum(distIJ[i, j] * y[i, j] for i in range(N) for j in open_j)
        + gp.quicksum(NOOPT_COST * y_noopt[i] for i in range(N))
        + ALPHA_WAIT * gp.quicksum(phi[j] for j in open_j),
        GRB.MINIMIZE,
    )
    for i in range(N):
        gm.addConstr(gp.quicksum(y[i, j] for j in open_j) + y_noopt[i] == d[i])
    for j in open_j:
        gm.addConstr(lam[j] == gp.quicksum(y[i, j] for i in range(N)))
        gm.addConstr(lam[j] <= float(ue_ev._cap[j, s_j[j] - 1]))
        si = s_j[j] - 1
        for k in range(ue_ev.N_bp):
            gm.addConstr(phi[j] >= float(ue_ev._F[j, si, k])
                         + float(ue_ev._w[j, si, k]) * (lam[j] - float(ue_ev._lambda_hat[j, si, k])))
    gm.update(); gm.optimize()

    lam_arr    = np.zeros(N)
    yn_arr     = np.array(d.copy())
    routed     = np.zeros(N)
    if gm.SolCount > 0:
        for j in open_j:
            lam_arr[j] = lam[j].X
        for i in range(N):
            yn_arr[i]  = y_noopt[i].X
            routed[i]  = 1.0 - yn_arr[i] / max(d[i], 1e-12)
    gm.dispose()
    return lam_arr, yn_arr, routed


# greedy scores

total_d  = float(np.sum(d))
U_greedy = greedy_u()
open_j   = sorted(j for (j, _), s in U_greedy.items() if s > 0)

grb_score, _ = grb_ev.evaluate(U_greedy)
ue_score,  _ = ue_ev.evaluate(U_greedy, T=T, cumulative_install=True)

print(f"instance   Vicenza  N=125  T=1  budget={BUDGET}  D={D_KM}km  Q={Q}  noopt={NOOPT_COST}km  alpha_wait={ALPHA_WAIT}")
print()
print(f"greedy     {len(open_j)} stations  {[(j, U_greedy[(j,0)]) for j in open_j]}")
print()
print("scores")
print(f"  GRB  {grb_score:.4f}  ({100*grb_score/total_d:.1f}%)")
print(f"  UE   {ue_score:.4f}  ({100*ue_score/total_d:.1f}%)")
print(f"  gap  {grb_score - ue_score:.4f}")
print()


# ranking

rng        = np.random.default_rng(42)
placements = [make_u(rng) for _ in range(N_RAND)]
grb_scores, ue_scores = [], []

for U in placements:
    gs, _ = grb_ev.evaluate(U)
    us, _ = ue_ev.evaluate(U, T=T, cumulative_install=True)
    grb_scores.append(gs)
    ue_scores.append(us)

grb_arr  = np.array(grb_scores)
ue_arr   = np.array(ue_scores)
grb_rank = np.argsort(np.argsort(-grb_arr)) + 1
ue_rank  = np.argsort(np.argsort(-ue_arr))  + 1

print(f"ranking  ({N_RAND} random placements)")
print(f"  {'#':>2}  {'open':>4}  {'GRB':>8}  {'UE':>8}  {'GRB-r':>6}  {'UE-r':>5}")
for i in range(N_RAND):
    n = sum(1 for s in placements[i].values() if s > 0)
    print(f"  {i+1:>2}  {n:>4}  {grb_scores[i]:>8.4f}  {ue_scores[i]:>8.4f}  {grb_rank[i]:>6}  {ue_rank[i]:>5}")

rho, pval = spearmanr(grb_scores, ue_scores)
print()
print(f"  Spearman rho={rho:.4f}  p={pval:.3f}")
if   rho >= 0.8: print("  strong agreement")
elif rho >= 0.5: print("  moderate agreement")
else:            print("  weak agreement")
print()


# extract allocations for greedy solution

# GRB: re-evaluate to refresh model solution
grb_ev.evaluate(U_greedy)
grb_assign = {}
for (i, j) in in_range:
    v = grb_ev._y.get((i, j, 0))
    if v is not None and v.X > 0.5:
        grb_assign[i] = j

# UE: full solve
lam_ue, yn_ue, routed_ue = solve_ue_full(U_greedy)

print("allocation extraction done — building plot")


# plot

import os
os.makedirs("results", exist_ok=True)

lon, lat  = coords_d[:, 0], coords_d[:, 1]   # display in degrees for readability
s_base    = 18 + 45 * (d / d.max())
covered   = set(grb_assign.keys())
uncovered = [i for i in range(N) if i not in covered]

GREY   = "#d0d0d0"
DARK   = "#333333"
RED    = "#c0392b"
BLUE   = "#2980b9"
station_colors = ["#e67e22", "#8e44ad", "#27ae60", "#2980b9"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="white")
fig.suptitle(
    f"Vicenza  N={N}  |  greedy placement  (budget={BUDGET}, D={D_KM}km, Q={Q})",
    fontsize=10, color=DARK, y=1.01,
)

for ax in axes:
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7, colors=DARK)
    ax.set_xlabel("lon", fontsize=7, color=DARK)
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")


# placement

ax = axes[0]
ax.scatter(lon, lat, s=s_base, c=GREY, linewidths=0, zorder=2)
for idx, j in enumerate(open_j):
    ax.scatter(lon[j], lat[j], s=120, c=station_colors[idx],
               edgecolors="white", linewidths=1.2, zorder=5)
    ax.annotate(f"{j}", (lon[j], lat[j]), fontsize=6, color=DARK,
                xytext=(3, 3), textcoords="offset points")
ax.set_title("placement", fontsize=8, color=DARK, pad=6)
ax.set_ylabel("lat", fontsize=7, color=DARK)

handles = [
    plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=station_colors[k],
               markersize=6, label=f"stn {j}  s={U_greedy[(j,0)]}")
    for k, j in enumerate(open_j)
]
ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper left")


# GRB coverage

ax = axes[1]
ax.scatter(lon[uncovered], lat[uncovered],
           s=[s_base[i] for i in uncovered],
           c=GREY, linewidths=0, zorder=2)

for idx, j in enumerate(open_j):
    assigned = [i for i, jj in grb_assign.items() if jj == j]
    if assigned:
        ax.scatter(lon[assigned], lat[assigned],
                   s=[s_base[i] for i in assigned],
                   c=station_colors[idx], alpha=0.75, linewidths=0, zorder=3)
    ax.scatter(lon[j], lat[j], s=120, c=station_colors[idx],
               edgecolors="white", linewidths=1.2, zorder=5)

ax.set_title(
    f"GRB   score {grb_score:.1f}  ({100*grb_score/total_d:.0f}%)\n"
    f"covered {len(covered)} / {N}   uncovered {len(uncovered)}",
    fontsize=8, color=DARK, pad=6,
)
ax.set_yticklabels([])


# UE routing fraction

ax = axes[2]
sc = ax.scatter(lon, lat, s=s_base, c=routed_ue, cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0, zorder=2)

lam_max = max(lam_ue.max(), 1e-9)
for idx, j in enumerate(open_j):
    cap_j = float(ue_ev._cap[j, U_greedy[(j, 0)] - 1])
    size  = 60 + 160 * lam_ue[j] / lam_max
    ax.scatter(lon[j], lat[j], s=size, c=station_colors[idx],
               edgecolors="white", linewidths=1.2, zorder=5)
    ax.annotate(f"{lam_ue[j]:.1f}/{cap_j:.1f}", (lon[j], lat[j]),
                fontsize=5.5, color=DARK, xytext=(4, 3), textcoords="offset points")

cb = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.03)
cb.set_label("routed fraction", fontsize=7)
cb.ax.tick_params(labelsize=6)

ax.set_title(
    f"UE   score {ue_score:.1f}  ({100*ue_score/total_d:.0f}%)\n"
    f"node colour = demand routed   circle size = load/cap",
    fontsize=8, color=DARK, pad=6,
)
ax.set_yticklabels([])

plt.tight_layout()
out = "results/vicenza_grb_vs_ue.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"plot saved -> {out}")
plt.show()

grb_ev.dispose()
