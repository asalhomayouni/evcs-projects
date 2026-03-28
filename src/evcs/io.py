from pathlib import Path
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist


def load_instance(csv_path, radius=0.02, T=8):
    """
    Load EVCS instance from CSV (like Bergamo dataset)

    Returns:
        inst dict with EVERYTHING needed for model + DR
    """

    csv_path = Path(csv_path)

    # =========================
    # 1) READ CSV
    # =========================
    df = pd.read_csv(csv_path)

    coords = df[["Centroid_Longitude", "Centroid_Latitude"]].values
    demand_vec = df["Aggregated_Population"].values

    N = len(coords)
    M = N  # assume demand nodes == candidate sites

    # =========================
    # 2) DISTANCES
    # =========================
    distIJ = cdist(coords, coords)

    # =========================
    # 3) COVERAGE (ARCS)
    # =========================
    in_range_matrix = (distIJ <= radius)

    Arcs = [(i, j) for i in range(N) for j in range(N) if in_range_matrix[i, j]]

    # adjacency lists
    Ji = {i: list(np.where(in_range_matrix[i])[0]) for i in range(N)}
    Ij = {j: list(np.where(in_range_matrix[:, j])[0]) for j in range(N)}

    # =========================
    # 4) DEMAND (MULTI-PERIOD)
    # =========================
    demand_IT = np.tile(demand_vec, (T, 1))  # shape (T, N)

    # =========================
    # 5) PACK INSTANCE
    # =========================
    inst = {
        "M": M,
        "N": N,
        "coords_I": coords,
        "coords_J": coords,
        "demand_I": demand_vec,
        "demand_IT": demand_IT,
        "distIJ": distIJ,
        "in_range": Arcs,   # IMPORTANT: arcs list for Pyomo
        "Ji": Ji,
        "Ij": Ij,
    }

    return inst