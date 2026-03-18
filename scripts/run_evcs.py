import time
from pathlib import Path
import pandas as pd
import sys

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path.home() / "scratch" / "evcs-projects"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(SRC_DIR))

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# =========================
# IMPORTS
# =========================
import Instance
import destruction_reconstruction as dr

from evcs.model import build_multi_period_model
from evcs.solve import solve_model

# =========================
# CHOOSE YOUR INSTANCE
# =========================
CSV_NAME = "input/center_40_Bergamo_k125.csv"

csv_path = PROJECT_ROOT / "data" / CSV_NAME

print(f"Loading instance: {csv_path}")

# =========================
# BUILD INSTANCE FROM CSV
# =========================
import numpy as np

df = pd.read_csv(csv_path)

coords = df[["Centroid_Longitude", "Centroid_Latitude"]].values
coords_I = coords.copy()
coords_J = coords.copy()
demand_vec = df["Aggregated_Population"].values

N = len(coords)
M = N   # assume all locations can host chargers (like your notebook)

print(f"N = {N}")

# Distance matrix
from scipy.spatial.distance import cdist
distIJ = cdist(coords, coords)

# Coverage radius (you can tune this later)
radius = 0.02  # 🔥 IMPORTANT (controls connectivity)

# Build in_range matrix
in_range = (distIJ <= radius).astype(int)

# 🔥 Convert to arc list for Pyomo
Arcs = [(i, j) for i in range(N) for j in range(N) if in_range[i, j] == 1]
# Build adjacency lists
Ji = {i: list(np.where(in_range[i] == 1)[0]) for i in range(N)}
Ij = {j: list(np.where(in_range[:, j] == 1)[0]) for j in range(N)}

# Multi-period demand
T = 8

demand_IT = np.tile(demand_vec, (T, 1))   # (T, N)
# Single-period proxy for DR
demand_I = demand_vec.copy()

# Pack into inst dict
inst = {
    "M": M,
    "N": N,
    "in_range": in_range,
    "Ji": Ji,
    "Ij": Ij,
    "demand_IT": demand_IT,
    "demand_I": demand_I,
    "distIJ": distIJ,
    "coords_I": coords_I,
    "coords_J": coords_J
}
# =========================
# PARAMETERS
# =========================
T = 8
P_T = [2, 2, 1, 2, 2, 1, 1, 1]

policy = "closest_priority"
Q = 12
max_chargers_per_site = 5

# DR
max_iter = 200
dr_time_limit = 300

# Exact
exact_time_limit = 1200

# =========================
# EXTRACT INSTANCE
# =========================
M = inst["M"]
N = inst["N"]
in_range = inst["in_range"]
Ji = inst["Ji"]
Ij = inst["Ij"]
demand = inst["demand_IT"]
dist = inst["distIJ"]

# =========================
# RUN
# =========================
print("Running DR...")
t0 = time.time()


# 🔥 Build DR-compatible demand (T, N)
demand_DR = inst["demand_I"].reshape(1, -1)   # (1, N)

inst_DR = inst.copy()
inst_DR["demand_IT"] = demand_DR   # shape (1, N)

dr_out = dr.run_DR_multi(
    inst=inst_DR,
    policy=policy,
    P_T=[sum(P_T)],   # collapse periods
    Q=Q,
    D=1.0,
    T=1,
    max_iter=max_iter,
    dr_time_limit=dr_time_limit,
    seed=11,
    max_chargers_per_site=max_chargers_per_site
)

dr_best = dr_out.get("best_obj", None)

print("Running Exact...")

model = build_multi_period_model(
    M=M,
    N=N,
    T=T,
    in_range=Arcs,
    Ji=Ji,
    Ij=Ij,
    demand_IT=demand,
    Q=Q,
    P_T=P_T,
    distIJ=dist,
    method_name=policy,
    allow_multi_charger=True,
    max_chargers_per_site=max_chargers_per_site
)
res = solve_model(
    model,
    time_limit=exact_time_limit
)

exact_obj = res.best_feasible_objective

t1 = time.time()

# =========================
# SAVE RESULTS
# =========================
gap = None
if dr_best and exact_obj:
    gap = (exact_obj - dr_best) / exact_obj

df = pd.DataFrame([{
    "instance": CSV_NAME,
    "DR_best": dr_best,
    "Exact_obj": exact_obj,
    "gap": gap,
    "runtime_s": t1 - t0
}])

outfile = RESULTS_DIR / f"result_{Path(CSV_NAME).stem}.csv"
df.to_csv(outfile, index=False)

print("Done!")
print(f"Saved to: {outfile}")
