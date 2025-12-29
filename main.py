import os
import networkx as nx
from quantum_sim.optimizer.hardware_quality_sweeper import HardwareQualitySweeper


def main():
    # Set thread count for Numba JIT operations
    os.environ["NUMBA_NUM_THREADS"] = "2"

    # Create a small graph for Max-Cut QAOA
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

    # Define hardware T1 ranges to sweep (in seconds)
    t1_sweep = [20e-6, 50e-6, 100e-6]

    # Initialize the sweeper
    sweeper = HardwareQualitySweeper(
        graph=graph,
        t1_range=t1_sweep,
        t2_to_t1_ratio=0.7,
        depolarizing_noise_prob=0.001,
        p_ex=0.005
    )

    # Run the sweep across circuit depths p=1 to p=4
    # F841 Fix: Removed unused assignment to sweep_results
    sweeper.run_sweep(max_p_layers=4, optimizer_maxiter=50)

    # Output results and save visualization
    sweeper.plot_sweep_results("qaoa_hardware_sweep.png")
    print("Optimization sweep completed successfully.")


if __name__ == "__main__":
    main()
