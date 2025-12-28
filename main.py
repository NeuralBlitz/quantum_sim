# quantum_sim/main.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

# Import core components
from quantum_sim.core.qubit import Qubit
from quantum_sim.core.register import Register
from quantum_sim.core.circuit import QuantumCircuit
from quantum_sim.core.parameter import Parameter
from quantum_sim.core.noise import DepolarizingChannel, ThermalRelaxationChannel

# Import gates
from quantum_sim.gates.single_qubit_gates import Hadamard, PauliX
from quantum_sim.gates.two_qubit_gates import CNOT
from quantum_sim.gates.parametric_gates import RX, RZ
from quantum_sim.gates.hadamard_block import HadamardBlock
from quantum_sim.gates.qaoa_cost_layer import QAOACostLayer
from quantum_sim.gates.qaoa_mixer_layer import QAOAMixerLayer

# Import backends
from quantum_sim.backends.numpy_backend import NumpyBackend
from quantum_sim.backends.qiskit_backend import QiskitBackend

# Import visualization
from quantum_sim.visualization.circuit_drawer import CircuitDrawer

# Import utility for expectation value
from quantum_sim.utils.expectation_value import ExpectationValueCalculator

# Import optimizer modules
from quantum_sim.optimizer.qaoa_optimizer import QAOAOptimizer
from quantum_sim.optimizer.sweet_spot_mapper import SweetSpotMapper, create_qaoa_ansatz_for_mapper
from quantum_sim.optimizer.hardware_quality_sweeper import HardwareQualitySweeper


# --- Graph Definition ---
def create_square_graph():
    G = nx.Graph()
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    return G

# --- Main QAOA Hardware Quality Sweep Execution ---
def run_qaoa_hardware_quality_sweep():
    print("--- Launching Final Expedition: QAOA Hardware Quality Sweep ---")

    graph = create_square_graph()
    num_qubits = graph.number_of_nodes()
    print(f"\nGraph defined with {num_qubits} nodes and {graph.number_of_edges()} edges:")
    print(graph.edges())

    # --- Define Hardware Tiers for the Sweep ---
    t1_tiers = [20e-6, 50e-6, 100e-6, 200e-6] # in seconds (20µs to 200µs)
    t2_ratio_to_t1 = 0.75 # T2 = 0.75 * T1, maintaining T2 <= T1
    depolarizing_p_global = 0.005 # 0.5% depolarizing error per gate (for all tiers in this sweep)

    sweeper = HardwareQualitySweeper(
        graph=graph,
        t1_range=t1_tiers,
        t2_to_t1_ratio=t2_ratio_to_t1,
        depolarizing_noise_prob=depolarizing_p_global,
        p_ex=0.0 # Assuming cold environment
    )

    max_p_layers_to_test = 5 # Test depths from p=1 to p=5
    optimizer_maxiter_per_p = 50 # Limited iterations for quick demo per p-value (can increase for more precision)
    
    _ = sweeper.run_sweep(max_p_layers=max_p_layers_to_test, optimizer_maxiter=optimizer_maxiter_per_p)

    # --- Generate the final publication-quality scientific visualization ---
    sweeper.plot_sweep_results(filename="hardware_quality_sweep_results.png")
    
    print("\n--- Hardware Quality Sweep Complete. Analysis generated. ---")


if __name__ == "__main__":
    num_cpu_threads = os.cpu_count()
    os.environ["NUMBA_NUM_THREADS"] = str(num_cpu_threads)
    print(f"Numba set to use {num_cpu_threads} threads for parallel execution (Kraus sum).")
    
    run_qaoa_hardware_quality_sweep()
