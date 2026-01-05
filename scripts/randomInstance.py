import os, json, pathlib, time
import numpy as np
import pandas as pd

DEFAULT_SEED = 42


# =========================================================
# Generate NEW Instance (default)
# =========================================================
def generate_instance(
    N=20,
    area=(0.0, 10.0, 0.0, 10.0),
    demand_low=1.0,
    demand_high=5.0,
    seed=DEFAULT_SEED
):
    rng = np.random.default_rng(seed)
    xmin, xmax, ymin, ymax = area

    xs = rng.uniform(xmin, xmax, size=N)
    ys = rng.uniform(ymin, ymax, size=N)
    coords = np.vstack([xs, ys]).T
    demand = rng.uniform(demand_low, demand_high, size=N)

    I_idx = list(range(N))
    J_idx = list(range(N))

    df = pd.DataFrame(coords, columns=["x", "y"])
    df["type"] = ""

    return dict(
        meta=dict(
            N=N,
            seed=seed,
            area=area,
            demand_low=demand_low,
            demand_high=demand_high,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        ),
        location_df=df,
        coords=coords,
        coords_I=coords[I_idx],
        coords_J=coords[J_idx],
        I_idx=I_idx,
        J_idx=J_idx,
        demand_I=demand.astype(float),
    )


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

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)

    # Optional: CSV
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

    # -----------------------------------------------------
    # OLD FORMAT (backward compatible)
    # -----------------------------------------------------
    else:
        coords = np.array(payload["coords"], dtype=float)
        demand = np.array(payload["demand_I"], dtype=float)

        I_idx = list(map(int, payload["I_idx"]))
        J_idx = list(map(int, payload["J_idx"]))

        meta = payload.get("meta", {})

    # Build DataFrame
    df = pd.DataFrame(coords, columns=["x", "y"])
    df["type"] = ""

    return dict(
        meta=meta,
        location_df=df,
        coords=coords,
        coords_I=coords[I_idx],
        coords_J=coords[J_idx],
        I_idx=I_idx,
        J_idx=J_idx,
        demand_I=demand,
    )


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

    inst = generate_instance(N=args.N, seed=args.seed)
    save_instance(inst, args.out.format(N=args.N, seed=args.seed))
