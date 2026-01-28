import os, json, pathlib, time
import numpy as np
import pandas as pd

DEFAULT_SEED = 42


def generate_instance(
    N=20,
    area=(0.0, 10.0, 0.0, 10.0),
    demand_low=1.0,
    demand_high=5.0,
    seed=DEFAULT_SEED,
    T=None,                     # number of periods (optional)
    demand_pattern="flat",       # "flat" | "trend_up" | "trend_down" | "seasonal"
    period_noise=0.05,           # small randomness per period

    # ----------------------------
    # NEW: harder demand controls
    # ----------------------------
    demand_dist="uniform",       # "uniform" | "lognormal"
    lognorm_sigma=1.0,           # tail heaviness (0.6 mild, 1.0 strong, 1.4 very heavy)
    n_hotspots=0,                # 0 disables hotspots; typical 1-5
    hotspot_strength=4.0,        # how strong hotspots are (e.g., 2-8)
    hotspot_radius=1.5,          # spatial radius in same units as coords
):
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = area

    xs = rng.uniform(xmin, xmax, size=N)
    ys = rng.uniform(ymin, ymax, size=N)
    coords = np.vstack([xs, ys]).T

    # ----------------------------
    # Base demand: uniform vs lognormal (heavy-tailed)
    # ----------------------------
    if str(demand_dist).lower() == "lognormal":
        raw = rng.lognormal(mean=0.0, sigma=float(lognorm_sigma), size=N)
        raw = raw / max(1e-12, raw.mean())  # normalize mean to 1
        scale = rng.uniform(demand_low, demand_high)
        demand = raw * scale
    else:
        demand = rng.uniform(demand_low, demand_high, size=N)

    # ----------------------------
    # Static spatial hotspots (do NOT move over time)
    # ----------------------------
    if int(n_hotspots) > 0:
        K = min(int(n_hotspots), N)
        center_idx = rng.choice(N, size=K, replace=False)
        centers = coords[center_idx]  # (K,2)

        boost = np.ones(N, dtype=float)
        r = float(hotspot_radius)
        s = float(hotspot_strength)

        # Gaussian bump around each hotspot center
        for c in centers:
            dist = np.sqrt(((coords - c) ** 2).sum(axis=1))
            boost += s * np.exp(-(dist ** 2) / (2.0 * r * r))

        demand = demand * boost

    # (optional) clip nonnegative
    demand = np.clip(demand, 0.0, None)

    I_idx = list(range(N))
    J_idx = list(range(N))

    df = pd.DataFrame(coords, columns=["x", "y"])
    df["type"] = ""

    inst = dict(
        meta=dict(
            N=N,
            seed=seed,
            area=area,
            demand_low=demand_low,
            demand_high=demand_high,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),

            # NEW meta (useful for reproducibility)
            demand_dist=str(demand_dist),
            lognorm_sigma=float(lognorm_sigma) if str(demand_dist).lower() == "lognormal" else None,
            n_hotspots=int(n_hotspots),
            hotspot_strength=float(hotspot_strength) if int(n_hotspots) > 0 else None,
            hotspot_radius=float(hotspot_radius) if int(n_hotspots) > 0 else None,
        ),
        location_df=df,
        coords=coords,
        coords_I=coords[I_idx],
        coords_J=coords[J_idx],
        I_idx=I_idx,
        J_idx=J_idx,
        demand_I=demand.astype(float),
    )

    # ----------------------------
    # Multi-period demand optional (STATIC hotspots: only global time factor + noise)
    # ----------------------------
    if T is not None:
        base = demand.astype(float)

        if demand_pattern == "flat":
            factors = np.ones(T)
        elif demand_pattern == "trend_up":
            factors = np.linspace(1.0, 1.3, T)
        elif demand_pattern == "trend_down":
            factors = np.linspace(1.0, 0.7, T)
        elif demand_pattern == "seasonal":
            factors = 1.0 + 0.2 * np.sin(np.linspace(0, 2*np.pi, T, endpoint=False))
        else:
            raise ValueError("demand_pattern must be flat|trend_up|trend_down|seasonal")

        demand_IT = []
        for t in range(T):
            noise = rng.normal(0.0, period_noise, size=N)
            d_t = base * factors[t] * (1.0 + noise)
            d_t = np.clip(d_t, 0.0, None)
            demand_IT.append(d_t.astype(float))

        inst["meta"]["T"] = int(T)
        inst["meta"]["demand_pattern"] = demand_pattern
        inst["meta"]["period_noise"] = float(period_noise)
        inst["demand_IT"] = demand_IT

    return inst



# =========================================================
# Save Instance — NEW Format
# =========================================================
def save_instance(inst, out_path):
    out_path = pathlib.Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": inst.get("meta", {}),
        "nodes": {
            "count": inst["meta"]["N"],
            "coords": [
                {"id": i, "x": float(x), "y": float(y)}
                for i, (x, y) in enumerate(inst["coords"])
            ],
            "demand": {str(i): float(d) for i, d in enumerate(inst["demand_I"])},
        },
        "indices": {
            "I_idx": list(inst["I_idx"]),
            "J_idx": list(inst["J_idx"]),
        },
    }

    # Add multi-period demand if present
    if "demand_IT" in inst:
        payload["nodes"]["demand_IT"] = [
            {str(i): float(d) for i, d in enumerate(inst["demand_IT"][t])}
            for t in range(len(inst["demand_IT"]))
        ]

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    # Optional: CSV (single-period demand only; you can extend if you want)
    csv_path = out_path.with_suffix(".csv")
    df = pd.DataFrame(inst["coords"], columns=["x", "y"])
    df["demand"] = inst["demand_I"]
    df.to_csv(csv_path, index_label="node_id")

    print(f"✅ Saved organized instance to {out_path}")
    print(f"   ↳ Also exported CSV to {csv_path}")


# =========================================================
# Load Instance — supports BOTH old + new formats
# =========================================================
def load_instance(in_path):
    in_path = pathlib.Path(in_path)

    with open(in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    demand_IT = None

    # -----------------------------------------------------
    # NEW FORMAT
    # -----------------------------------------------------
    if "nodes" in payload and "indices" in payload:
        coords = np.array(
            [[n["x"], n["y"]] for n in payload["nodes"]["coords"]],
            dtype=float
        )
        demand = np.array(
            [float(payload["nodes"]["demand"][str(i)]) for i in range(len(coords))]
        )

        I_idx = list(map(int, payload["indices"]["I_idx"]))
        J_idx = list(map(int, payload["indices"]["J_idx"]))

        meta = payload.get("meta", {})

        if "demand_IT" in payload["nodes"]:
            demand_IT = [
                np.array([float(payload["nodes"]["demand_IT"][t][str(i)]) for i in range(len(coords))], dtype=float)
                for t in range(len(payload["nodes"]["demand_IT"]))
            ]

    # -----------------------------------------------------
    # OLD FORMAT (backward compatible)
    # -----------------------------------------------------
    else:
        coords = np.array(payload["coords"], dtype=float)
        demand = np.array(payload["demand_I"], dtype=float)

        I_idx = list(map(int, payload["I_idx"]))
        J_idx = list(map(int, payload["J_idx"]))

        meta = payload.get("meta", {})

        # Old format might have demand_IT directly
        if "demand_IT" in payload:
            demand_IT = [np.array(x, dtype=float) for x in payload["demand_IT"]]

    df = pd.DataFrame(coords, columns=["x", "y"])
    df["type"] = ""

    inst = dict(
        meta=meta,
        location_df=df,
        coords=coords,
        coords_I=coords[I_idx],
        coords_J=coords[J_idx],
        I_idx=I_idx,
        J_idx=J_idx,
        demand_I=demand,
    )
    if demand_IT is not None:
        inst["demand_IT"] = demand_IT
    return inst


# =========================================================
# CLI (optional)
# =========================================================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="evcs-projects/data/random_instances/inst_N{N}_seed{seed}.json")
    args = ap.parse_args()

    inst = generate_instance(
        N=args.N,
        seed=args.seed,
        # Example harder settings (edit as you like):
        demand_dist="lognormal",
        lognorm_sigma=1.0,
        n_hotspots=3,
        hotspot_strength=5.0,
        hotspot_radius=1.2,
    )
    save_instance(inst, args.out.format(N=args.N, seed=args.seed))
