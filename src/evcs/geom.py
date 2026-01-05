import numpy as np
from scipy.spatial.distance import cdist

def build_arcs(coords_I, coords_J, D, *, forbid_self=True, I_idx=None, J_idx=None):
    coords_I = np.asarray(coords_I, float)
    coords_J = np.asarray(coords_J, float)
    distIJ = cdist(coords_I, coords_J, metric="euclidean")

    M, N = distIJ.shape
    in_range = []
    Ji = {i: [] for i in range(M)}
    Ij = {j: [] for j in range(N)}

    for i in range(M):
        for j in range(N):
            if distIJ[i, j] <= D:
                if forbid_self and (I_idx is not None) and (J_idx is not None) and I_idx[i] == J_idx[j]:
                    continue
                in_range.append((i, j))
                Ji[i].append(j)
                Ij[j].append(i)

    return distIJ, in_range, Ji, Ij

def compute_farther(distIJ, in_range, Ji):
    farther_of = {}
    for (i, j) in in_range:
        d_ij = distIJ[i, j]
        farther = [jp for jp in Ji[i] if distIJ[i, jp] > d_ij]
        if farther:
            farther_of[(i, j)] = farther
    return farther_of
import numpy as np

def pairwise_dist(coords_I, coords_J):
    coords_I = np.asarray(coords_I)
    coords_J = np.asarray(coords_J)
    dists = np.sqrt(
        (coords_I[:, np.newaxis, 0] - coords_J[np.newaxis, :, 0]) ** 2 +
        (coords_I[:, np.newaxis, 1] - coords_J[np.newaxis, :, 1]) ** 2
    )
    return dists
