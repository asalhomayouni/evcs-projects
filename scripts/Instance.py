# randomInstance_fixed.py
# ------------------------------------------------------------
# Random / semi-realistic instance generator for EVCS multi-period
# Outputs:
#   inst["coords_I"]   : (M,2) demand nodes
#   inst["coords_J"]   : (N,2) candidate sites
#   inst["demand_IT"]  : (T,M) demand by period t at node i
#   inst["meta"]       : dict of generation parameters
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(None if seed is None else int(seed))


def _pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: (m,2), B: (n,2) -> dist: (m,n)
    diff = A[:, None, :] - B[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _seasonal_multiplier(T: int, pattern: str) -> np.ndarray:
    pattern = (pattern or "flat").lower()
    t = np.arange(T, dtype=float)

    if pattern == "flat":
        mult = np.ones(T, dtype=float)
    elif pattern == "trend_up":
        # mild upward trend
        mult = 0.85 + 0.30 * (t / max(1.0, T - 1.0))
    elif pattern == "trend_down":
        mult = 1.15 - 0.30 * (t / max(1.0, T - 1.0))
    elif pattern == "seasonal":
        # one sine wave over horizon, centered at 1.0
        mult = 1.0 + 0.20 * np.sin(2.0 * np.pi * t / max(1.0, T))
        mult = np.clip(mult, 0.60, 1.40)
    else:
        # fallback
        mult = np.ones(T, dtype=float)

    return mult


def _base_demand_vector(
    M: int,
    rng: np.random.Generator,
    demand_dist: str,
    lognorm_sigma: float,
) -> np.ndarray:
    demand_dist = (demand_dist or "lognormal").lower()
    if demand_dist == "uniform":
        base = rng.uniform(0.5, 1.5, size=M)
    elif demand_dist == "lognormal":
        # median ~1.0, heavy tail controlled by sigma
        sigma = float(lognorm_sigma)
        mu = -0.5 * sigma * sigma  # so E[lognormal] ~ 1 if desired-ish
        base = rng.lognormal(mean=mu, sigma=sigma, size=M)
        # keep extreme tails bounded (optional)
        base = np.clip(base, 0.05, np.quantile(base, 0.995))
    else:
        base = rng.uniform(0.5, 1.5, size=M)

    # normalize to mean ~1 for stability across knobs
    base = base / max(1e-12, float(np.mean(base)))
    return base


def _apply_hotspots(
    coords_I: np.ndarray,
    base: np.ndarray,
    rng: np.random.Generator,
    hotspots: int,
    hotspot_strength: float,
    hotspot_radius: float,
) -> np.ndarray:
    base2 = base.copy()
    hotspots = int(hotspots)

    if hotspots <= 0:
        return base2

    strength = float(hotspot_strength)
    radius = float(hotspot_radius)

    # pick hotspot centers among demand nodes
    centers_idx = rng.choice(len(coords_I), size=min(hotspots, len(coords_I)), replace=False)
    centers = coords_I[centers_idx]

    # for each center, boost nearby nodes with gaussian-like bump
    for c in centers:
        d = np.sqrt(np.sum((coords_I - c) ** 2, axis=1))
        bump = np.exp(-(d ** 2) / (2.0 * (radius ** 2 + 1e-12)))
        base2 *= (1.0 + (strength - 1.0) * bump)

    # renormalize mean ~1
    base2 = base2 / max(1e-12, float(np.mean(base2)))
    return base2


def generate_instance(
    N_sites: int = 300,
    T: int = 6,
    seed: int = 11,
    # demand side size:
    same_IJ: bool = True,          # if True, demand nodes == candidate sites
    M_demand: Optional[int] = None, # if same_IJ=False, specify M separately
    # geometry
    area_scale: float = 1.0,        # coordinates in [0, area_scale]
    # demand time pattern
    demand_pattern: str = "seasonal",
    period_noise: float = 0.2,
    # demand distribution hardness
    demand_dist: str = "lognormal",
    lognorm_sigma: float = 1.3,
    # hotspots
    hotspots: int = 5,
    hotspot_strength: float = 10.0,
    hotspot_radius: float = 0.6,
) -> Dict[str, Any]:
    """
    Returns inst dict with:
      coords_I (M,2), coords_J (N,2), demand_IT (T,M), meta dict.
    """

    rng = _rng(seed)
    N = int(N_sites)
    T = int(T)
    area_scale = float(area_scale)

    # Candidate sites
    coords_J = rng.uniform(0.0, area_scale, size=(N, 2))

    # Demand nodes
    if same_IJ:
        coords_I = coords_J.copy()
    else:
        M = int(M_demand) if M_demand is not None else max(50, int(0.8 * N))
        coords_I = rng.uniform(0.0, area_scale, size=(M, 2))

    M = int(coords_I.shape[0])

    # Base demand per node
    base = _base_demand_vector(M, rng, demand_dist=demand_dist, lognorm_sigma=lognorm_sigma)

    # Hotspots (optional)
    base = _apply_hotspots(
        coords_I=coords_I,
        base=base,
        rng=rng,
        hotspots=hotspots,
        hotspot_strength=hotspot_strength,
        hotspot_radius=hotspot_radius,
    )

    # Time multipliers + noise
    mult = _seasonal_multiplier(T, demand_pattern)
    noise = float(period_noise)

    demand_IT = np.zeros((T, M), dtype=float)
    for t in range(T):
        eps = rng.normal(loc=0.0, scale=noise, size=M)
        # keep demand positive
        d = base * mult[t] * (1.0 + eps)
        d = np.clip(d, 0.0, None)
        demand_IT[t, :] = d

    # Optional normalization (keeps totals stable across knobs)
    mean_total = float(np.mean(np.sum(demand_IT, axis=1)))
    if mean_total > 0:
        demand_IT *= (M / mean_total)  # average total per period ~ M

    inst = {
        "coords_I": coords_I.astype(float),
        "coords_J": coords_J.astype(float),
        "demand_IT": demand_IT.astype(float),  # (T,M)
        "meta": {
            "N_sites": N,
            "M_demand": M,
            "T": T,
            "seed": int(seed),
            "same_IJ": bool(same_IJ),
            "area_scale": float(area_scale),
            "demand_pattern": str(demand_pattern),
            "period_noise": float(period_noise),
            "demand_dist": str(demand_dist),
            "lognorm_sigma": float(lognorm_sigma),
            "hotspots": int(hotspots),
            "hotspot_strength": float(hotspot_strength),
            "hotspot_radius": float(hotspot_radius),
        }
    }
    return inst


def make_instance_name(inst: Dict[str, Any]) -> str:
    meta = inst.get("meta", {})
    N = meta.get("N_sites", inst["coords_J"].shape[0])
    T = meta.get("T", inst["demand_IT"].shape[0])
    seed = meta.get("seed", 0)
    pattern = meta.get("demand_pattern", "flat")
    noise = meta.get("period_noise", 0.0)
    dist = meta.get("demand_dist", "lognormal")
    sig = meta.get("lognorm_sigma", 1.0)
    hot = meta.get("hotspots", 0)
    hs = meta.get("hotspot_strength", 1.0)
    hr = meta.get("hotspot_radius", 0.0)

    return (
        f"inst_N{N}_seed{seed}_T{T}_"
        f"{pattern}_noise{noise:.2f}_"
        f"{dist}_sig{sig:.2f}_"
        f"hot{hot}_s{hs:.1f}_r{hr:.2f}"
    )


def save_instance_npz(inst: Dict[str, Any], path: str) -> None:
    """
    Save in npz with arrays + serialized meta.
    """
    meta = inst.get("meta", {})
    np.savez_compressed(
        path,
        coords_I=inst["coords_I"],
        coords_J=inst["coords_J"],
        demand_IT=inst["demand_IT"],
        meta=np.array([meta], dtype=object),
    )


def load_instance_npz(path: str) -> Dict[str, Any]:
    """
    Load from npz created by save_instance_npz.
    """
    z = np.load(path, allow_pickle=True)
    meta_arr = z.get("meta", None)
    meta = {}
    if meta_arr is not None and len(meta_arr) > 0:
        meta = dict(meta_arr[0])

    return {
        "coords_I": z["coords_I"],
        "coords_J": z["coords_J"],
        "demand_IT": z["demand_IT"],
        "meta": meta,
    }