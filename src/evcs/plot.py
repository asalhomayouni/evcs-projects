# evcs/plot.py
import math
from typing import Sequence, Dict, Tuple
import matplotlib.pyplot as plt
from pyomo.environ import value as pyo_value


def _v(x) -> float:
    """Safe float(value(x)) with None -> 0.0."""
    try:
        vx = pyo_value(x)
        return 0.0 if vx is None else float(vx)
    except Exception:
        return 0.0


def plot_nodes(location_df, I_idx: Sequence[int] = None, J_idx: Sequence[int] = None) -> None:
    """Scatter all points; highlight demand set I and site set J."""
    plt.scatter(location_df["x"], location_df["y"], s=16, c="lightgray", label="All")

    if I_idx is not None and len(I_idx) > 0:
        plt.scatter(
            location_df.loc[I_idx, "x"],
            location_df.loc[I_idx, "y"],
            s=18, c="green", label="Demand (I)"
        )

    if J_idx is not None and len(J_idx) > 0:
        plt.scatter(
            location_df.loc[J_idx, "x"],
            location_df.loc[J_idx, "y"],
            s=80, marker="^", facecolors="none", edgecolors="red", label="Sites (J)"
        )

    plt.xlabel("X"); plt.ylabel("Y"); plt.grid(True); plt.legend()


def plot_opened_sites(location_df, opened_rows: Sequence[int]) -> None:
    """Overlay opened sites (original row indices)."""
    for k, r in enumerate(opened_rows):
        plt.scatter(
            location_df.iloc[r].x, location_df.iloc[r].y,
            s=110, marker="^", c="red", label="Opened CS" if k == 0 else ""
        )


def plot_solution_pretty(
    m,
    location_df,
    coords_all,
    I_idx: Sequence[int],
    J_idx: Sequence[int],
    in_range: Sequence[Tuple[int, int]],
    show_self: str = "ring",  
    eps: float = 1e-6
) -> None:
    """
    Pretty plot:
      - grey = all nodes
      - green = served demand nodes
      - red triangles = opened stations
      - green lines between demand and station (for every y[i,j] > eps)
      - distance label on each line
      - 'f=..' (sum_j y[i,j]) next to each served demand node
    """
    # which stations are opened 
    opened_J = [j for j in m.J if _v(m.x[j]) > 0.5]
    opened_rows = [J_idx[j] for j in opened_J]

    assign: Dict[int, list] = {}   
    frac_i: Dict[int, float] = {} 
    for i in m.I:
        s = 0.0
        for j in m.J:
            if (i, j) in in_range:
                yij = _v(m.y[i, j])
                if yij > eps:
                    assign.setdefault(i, []).append((j, yij))
                    s += yij
        if s > eps:
            frac_i[i] = min(1.0, s)

    served_I = sorted(frac_i.keys())
    uncovered_I = [i for i in m.I if i not in served_I]

    # base: all nodes grey
    plt.scatter(location_df["x"], location_df["y"], s=18, c="lightgray", label="All nodes")

    # opened stations (red triangles)
    for k, r in enumerate(opened_rows):
        plt.scatter(location_df.iloc[r].x, location_df.iloc[r].y,
                    s=110, marker="^", c="red", label="Opened CS" if k == 0 else "")

    # served demand nodes (green dots)
    if served_I:
        sx = [coords_all[I_idx[i]][0] for i in served_I]
        sy = [coords_all[I_idx[i]][1] for i in served_I]
        plt.scatter(sx, sy, s=24, c="green", label="Served demand")

    # draw assignments (green lines) + distance labels
    for i in served_I:
        xi, yi = location_df.loc[I_idx[i], ["x", "y"]]
        for j, yij in assign[i]:
            j_row = J_idx[j]
            xj, yj = location_df.loc[j_row, ["x", "y"]]

        
            if I_idx[i] == j_row:
                if show_self == "ring":
                    plt.scatter([xi], [yi], s=140, facecolors="none",
                                edgecolors="green", alpha=0.5)
                elif show_self == "tick":
                    plt.plot([xi, xi + 2.0], [yi, yi + 2.0],
                             color="green", alpha=0.7, linewidth=1.5)
                continue

            # edge
            plt.plot([xi, xj], [yi, yj], color="green", alpha=0.45, linewidth=1.6)

            # true distance label at midpoint
            dij = math.hypot(float(xi - xj), float(yi - yj))
            mx, my = (xi + xj) / 2.0, (yi + yj) / 2.0
            plt.annotate(f"{dij:.2f}", (mx, my),
                         textcoords="offset points", xytext=(0, 0),
                         ha="center", va="center", fontsize=8, color="darkgreen")

    # show each served demand with total fraction met
    for i in served_I:
        xi, yi = location_df.loc[I_idx[i], ["x", "y"]]
        plt.annotate(f"f={frac_i[i]:.2f}", (xi, yi),
                     textcoords="offset points", xytext=(6, 6),
                     fontsize=9, color="black")

    # mark uncovered demand nodes
    for i in uncovered_I:
        xi, yi = location_df.loc[I_idx[i], ["x", "y"]]
        plt.annotate(f"{I_idx[i]} (uncovered)", (xi, yi),
                     textcoords="offset points", xytext=(6, -10),
                     fontsize=8, color="gray")

    plt.title("Assignments with distances and per-node fraction served")
    plt.axis("equal"); plt.xlabel("X"); plt.ylabel("Y")
    plt.grid(True); plt.legend()
def plot_assignments_verbose(*args, **kwargs):
    return plot_solution_pretty(*args, **kwargs)
