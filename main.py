# quantum_sim/main.py

import os
import numpy as np
import networkx as nx

# Import core components
from quantum_sim.core.register import Register
from quantum_sim.core.circuit import QuantumCircuit

# Import optimizer modules
from quantum_sim.optimizer.hardware_quality_sweeper import HardwareQualitySweeper


def create_square_graph():
    """
    Creates a simple 4-node square graph for Max-Cut testing.
    """
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    return graph


def run_qaoa_hardware_quality_sweep():
    """
    Executes a sweep across different T1/T2 tiers to find the p-migration effect.
    """
    print("--- Launching Final Expedition: QAOA Hardware Quality Sweep ---")

    graph = create_square_graph()
    num_qubits = graph.number_of_nodes()
    print(f"\nGraph defined with {num_qubits} nodes and {graph.number_of_edges()} edges:")
    print(graph.edges())

    # --- Define Hardware Tiers for the Sweep ---
    t1_tiers = [20e-6, 50e-6, 100e-6, 200e-6]  # in seconds (20µs to 200µs)
    t2_ratio_to_t1 = 0.75  # T2 = 0.75 * T1
    depolarizing_p_global = 0.005  # 0.5% depolarizing error per gate

    sweeper = HardwareQualitySweeper(
        graph=graph,
        t1_range=t1_tiers,
        t2_to_t1_ratio=t2_ratio_to_t1,
        depolarizing_noise_prob=depolarizing_p_global,
        p_ex=0.0  # Assuming cold environment
    )

    max_p_layers_to_test = 5
    optimizer_maxiter_per_p = 50
    
    _ = sweeper.run_sweep(
        max_p_layers=max_p_layers_to_test, 
        optimizer_maxiter=optimizer_maxiter_per_p
    )

    # --- Generate the final publication-quality scientific visualization ---
    sweeper.plot_sweep_results(filename="hardware_quality_sweep_results.png")
    
    print("\n--- Hardware Quality Sweep Complete. Analysis generated. ---")


if __name__ == "__main__":
    # Prioritize CI threads if set, otherwise use system count
    num_threads = os.getenv("NUMBA_NUM_THREADS", str(os.cpu_count()))
    os.environ["NUMBA_NUM_THREADS"] = num_threads
    print(f"Numba set to use {num_threads} threads for parallel Kraus sum execution.")
    
    run_qaoa_hardware_quality_sweep()
