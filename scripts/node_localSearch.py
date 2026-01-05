import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from evcs.model import build_base_model
from evcs.methods import (
    build_initial_solution,
    apply_method,
    evaluate_solution,
    compute_farther,
    local_search,
)
from evcs.solve import solve_model
from evcs.geom import build_arcs
from evcs.plot import plot_solution_pretty, plot_allowed_arcs

# Layout for 6-node test system

def make_6node_layout():
    coords = np.array([
        [0.0, -1.0],
        [0.0,  1.0],
        [1.0,  0.0],
        [2.5,  0.0],
        [4.5,  0.0],
        [5.5,  1.0],
    ], dtype=float)

    demand = np.array([2.0, 1.0, 3.0, 5.0, 4.0, 1.0], dtype=float)

    location_df = pd.DataFrame(coords, columns=["x", "y"])
    location_df["type"] = ""
    I_idx = list(range(6))
    J_idx = list(range(6))

    return dict(
        location_df=location_df,
        coords=coords,
        coords_I=coords[I_idx],
        coords_J=coords[J_idx],
        I_idx=I_idx,
        J_idx=J_idx,
        demand_I=demand,
    )


#Run the experiment
def run_policy(method_name, P=2, Q=5.0, D=2.0, forbid_self=False, show=True):
    print(f"\n===============================================")
    print(f" Running Allocation Policy: {method_name}")
    print(f"===============================================")

    #Setup instance
    inst = make_6node_layout()
    M, N = len(inst["I_idx"]), len(inst["J_idx"])
    distIJ, in_range, Ji, Ij = build_arcs(
        inst["coords_I"], inst["coords_J"], D=D,
        forbid_self=forbid_self, I_idx=inst["I_idx"], J_idx=inst["J_idx"]
    )

    #Build base model
    m = build_base_model(M, N, in_range, Ji, Ij, inst["demand_I"], Q, P,
                         distIJ=distIJ, method_name="none")

    #tep 1: Initial greedy allocation
    m_init = build_initial_solution(m, distIJ, mode="greedy")
    solve_model(m_init, verbose=False)
    init_metrics = evaluate_solution(m_init, distIJ, inst["demand_I"])

    print("\n🟡 INITIAL (Greedy) SOLUTION")
    print(f"Covered demand = {init_metrics['covered_demand']:.2f} "
          f"({init_metrics['covered_pct']:.1f}%)")
    print(f"Total distance = {init_metrics['total_distance']:.3f}")
    print(f"Value score    = {init_metrics['score']:.3f}")

    if show:
        plt.figure(figsize=(7, 6))
        plot_allowed_arcs(inst["location_df"], inst["I_idx"], inst["J_idx"], in_range, alpha=0.25)
        plot_solution_pretty(m_init, inst["location_df"], inst["coords"],
                             inst["I_idx"], inst["J_idx"], in_range)
        plt.title(f"Initial Greedy Allocation ({method_name})")
        plt.show()

    #Step 2: Apply allocation policy
    farther_of = compute_farther(distIJ, in_range, Ji)
    m_policy = apply_method(m_init, method_name, distIJ, in_range, Ji, Ij, farther_of)

    #Step 3: Run local search only for valid cases
    if method_name in ["local_search", "closest_only", "closest_priority", "system_optimum"]:
        print("\n🔴 Starting Local Search Improvement...")
        m_best = local_search(
            m_policy, distIJ, in_range, Ji, Ij, farther_of,
            method_name=method_name, max_iter=30
        )
        best_metrics = evaluate_solution(m_best, distIJ, inst["demand_I"])
    else:
        print("\n⚙️  This method does not require local search (static allocation).")
        m_best = m_policy
        best_metrics = evaluate_solution(m_best, distIJ, inst["demand_I"])

    #Step 4: Print results
    print("\n🟢 FINAL BEST SOLUTION")
    print(f"Covered demand = {best_metrics['covered_demand']:.2f} "
          f"({best_metrics['covered_pct']:.1f}%)")
    print(f"Total distance = {best_metrics['total_distance']:.3f}")
    print(f"Value score    = {best_metrics['score']:.3f}")
    print(f"Improvement    = "
          f"{init_metrics['total_distance'] - best_metrics['total_distance']:.3f}")

    #Step 5: Plot final result
    if show:
        plt.figure(figsize=(7, 6))
        plot_allowed_arcs(inst["location_df"], inst["I_idx"], inst["J_idx"], in_range, alpha=0.25)
        plot_solution_pretty(m_best, inst["location_df"], inst["coords"],
                             inst["I_idx"], inst["J_idx"], in_range)
        plt.title(f"Best Solution After Local Search ({method_name})")
        plt.show()

    #Return dictionary
    return dict(
    method=method_name,
    model=m_best,                
    distIJ=distIJ,              
    init_metrics=init_metrics,
    best_metrics=best_metrics,
)