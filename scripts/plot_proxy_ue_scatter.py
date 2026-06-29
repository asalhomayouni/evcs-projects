"""
plot_proxy_ue_scatter.py

Collects (binary-proxy, UE-score) pairs from ALNS candidates on Monza N=400
and produces a thesis-quality scatter figure.

Usage:
    python scripts/plot_proxy_ue_scatter.py
    python scripts/plot_proxy_ue_scatter.py --n-samples 200 --max-iter 400
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT / "src"))

from evcs.geom import build_arcs
from evcs.destroy import destroy_multi_u
from evcs.greedy import reconstruct_u_dict_fast
from evcs.methods import greedy_schedule_multi_from_variants
from evcs.ue_evaluator import UEEvaluator
from evcs.proxy import evaluate_u_numpy_greedy_binary

# ── CLI ───────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--csv",           default="center_79_Monza_k400.csv")
p.add_argument("--seed",          type=int,   default=11)
p.add_argument("--T",             type=int,   default=6)
p.add_argument("--D",             type=float, default=2.0)
p.add_argument("--n-samples",     type=int,   default=50,
               help="Number of (proxy, UE) pairs to collect")
p.add_argument("--div-seed",      type=int,   default=421,
               help="RNG seed for diversity (destroy-reconstruct)")
p.add_argument("--ue-alpha-wait",       type=float, default=10.0)
p.add_argument("--ue-noopt",            type=float, default=30.0)
p.add_argument("--ue-max-range",        type=float, default=2.0)
p.add_argument("--ue-penalty-distance", type=float, default=5.0)
p.add_argument("--ue-cap-per-server",   type=float, default=4.0)
args = p.parse_args()

OUT_DIR = PROJECT_ROOT / "results" / "diagnostics" / "proxy_scatter"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = PROJECT_ROOT / "data" / "input"

MAX_CHARGERS  = 6
Q_CAP         = 20.0
CUMULATIVE    = True
DESTROY_MODES = ["site_swap", "local_remove", "area_destroy"]
LS_MODES      = ("site_swap", "local_remove")

# ── Load instance ─────────────────────────────────────────────────────────────
df_raw     = pd.read_csv(DATA_DIR / args.csv)
coords_deg = df_raw[["Centroid_Longitude", "Centroid_Latitude"]].to_numpy(float)
coords_km  = coords_deg * 111.0

pop        = np.maximum(df_raw["Aggregated_Population"].to_numpy(float), 0.0)
pop_share  = pop / pop.sum()
M          = coords_km.shape[0]

_period_budget = 8   # matches original diagnostic run (48 total chargers, P_T=[8]*6)
P_T = [_period_budget] * args.T

base     = pop_share * M
rng_inst = np.random.default_rng(args.seed)
demand_TM = np.zeros((args.T, M))
for t in range(args.T):
    sf = 1.0 + 0.2 * np.sin(2 * np.pi * t / args.T)
    demand_TM[t] = np.maximum(0.0, base * sf * (1.0 + rng_inst.normal(0, 0.05, M)))

distIJ, in_range, Ji, Ij = build_arcs(coords_km, coords_km, D=args.D, forbid_self=False)

Ij_int = {j: [] for j in range(M)}
Ji_int = {i: [] for i in range(M)}
for (i, j) in in_range:
    Ij_int[j].append(i)
    Ji_int[i].append(j)

J_i_list = [sorted(Ji_int[i], key=lambda j: distIJ[i, j]) for i in range(M)]

inst = {
    "coords_I": coords_km, "coords_J": coords_km.copy(),
    "demand_IT": demand_TM, "distIJ": distIJ,
    "in_range": in_range, "Ji": Ji, "Ij": Ij, "M": M, "N": M,
}

print(f"Instance : {args.csv}")
print(f"N={M}  T={args.T}  D={args.D}km  budget={sum(P_T)} chargers")

# ── UEEvaluator ───────────────────────────────────────────────────────────────
ue_mu = args.ue_cap_per_server + 1e-4
ue_eval = UEEvaluator(
    N=M, d=demand_TM[0], tau=distIJ, mu=ue_mu, s_max=MAX_CHARGERS,
    noopt_cost=args.ue_noopt, alpha_wait=args.ue_alpha_wait, N_bp=50,
    max_range=args.ue_max_range, penalty_distance=args.ue_penalty_distance,
)

def proxy_fn(U):
    return evaluate_u_numpy_greedy_binary(
        U, demand_TM=demand_TM, J_i_list=J_i_list, distIJ=distIJ,
        Q_cap=Q_CAP, T=args.T, N=M, cumulative_install=CUMULATIVE,
        pre_sorted_J_i=J_i_list,
    )

def ue_fn(U):
    score, _ = ue_eval.evaluate(U, T=args.T, cumulative_install=CUMULATIVE)
    return float(score)

# ── Data collection: diverse destroy-reconstruct from greedy baseline ─────────
# Replicates the original diagnostic that produced r ≈ 0.92 (P_T=[8]*6, noopt=30,
# random frac 0.3-0.7, div_seed=29, no local search step).
rng = np.random.default_rng(args.div_seed)

U_base = greedy_schedule_multi_from_variants(inst, P_T, args.D, seed=args.seed)

pairs_proxy = []
pairs_ue    = []

print(f"\nCollecting {args.n_samples} (proxy, UE) pairs via diverse D&R ...")

for k in range(args.n_samples):
    frac = float(rng.uniform(0.3, 0.7))
    mode = str(rng.choice(DESTROY_MODES))

    U_partial, _ = destroy_multi_u(
        U_base, inst, rng, P_T, frac, mode,
        site_cap=MAX_CHARGERS, cumulative_install=CUMULATIVE,
    )
    U_recon, _ = reconstruct_u_dict_fast(
        U_partial, demand_TM, P_T, Ij_int,
        U_cap=MAX_CHARGERS, Q=Q_CAP, rng=rng,
        cumulative_install=CUMULATIVE,
    )

    px = proxy_fn(U_recon)
    ue = ue_fn(U_recon)
    pairs_proxy.append(px)
    pairs_ue.append(ue)

    if (k + 1) % 10 == 0:
        print(f"  {k+1:>3}/{args.n_samples}  proxy={px:.1f}  UE={ue:.3f}")

ue_eval.dispose()

x = np.array(pairs_proxy[:args.n_samples])
y = np.array(pairs_ue[:args.n_samples])
r, _ = stats.pearsonr(x, y)

print(f"\nCollected {len(x)} pairs  |  Pearson r = {r:.4f}")

# Save the data
df_pairs = pd.DataFrame({"binary_proxy": x, "ue_score": y})
csv_out = OUT_DIR / "proxy_ue_pairs_monza.csv"
df_pairs.to_csv(csv_out, index=False)
print(f"Data saved: {csv_out}")

# ── Thesis-quality figure ─────────────────────────────────────────────────────
slope, intercept, *_ = stats.linregress(x, y)
x_line = np.linspace(x.min(), x.max(), 200)
y_line = slope * x_line + intercept

plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.linewidth":    0.8,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "grid.alpha":        0.30,
    "grid.linestyle":    "--",
    "grid.linewidth":    0.5,
    "legend.frameon":    False,
})

fig, ax = plt.subplots(figsize=(5.5, 4.5))

ax.scatter(x, y,
           s=28, color="#3B82C4", alpha=0.65,
           linewidths=0.4, edgecolors="white",
           zorder=3, label="Candidate solutions")

ax.plot(x_line, y_line,
        color="#C0392B", linewidth=1.5,
        linestyle="--", zorder=4, label="OLS regression")

ax.grid(axis="both", zorder=0)

ax.set_xlabel(r"Binary proxy score  ($Q_\mathrm{cap} = 20$)", fontsize=11)
ax.set_ylabel("UE score  (full evaluation)", fontsize=11)

# Clean r annotation — bottom-right, no box
ax.text(0.97, 0.06,
        f"$r = {r:.2f}$",
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=12, color="#C0392B")

ax.tick_params(labelsize=9)

# Subtle x/y range padding
pad_x = (x.max() - x.min()) * 0.04
pad_y = (y.max() - y.min()) * 0.08
ax.set_xlim(x.min() - pad_x, x.max() + pad_x)
ax.set_ylim(y.min() - pad_y, y.max() + pad_y)

ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

fig.tight_layout()

out_pdf = OUT_DIR / "proxy_ue_scatter_thesis.pdf"
out_png = OUT_DIR / "proxy_ue_scatter_thesis.png"
fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
fig.savefig(out_png, dpi=200, bbox_inches="tight")
plt.close(fig)
plt.rcdefaults()

print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")
print(f"Final:  N={len(x)}  r={r:.4f}")
print("Done.")
