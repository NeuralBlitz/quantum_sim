import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any

from quantum_sim.optimizer.qaoa_optimizer import QAOAOptimizer
from quantum_sim.core.circuit import QuantumCircuit
from quantum_sim.core.parameter import Parameter
from quantum_sim.core.noise import DepolarizingChannel
from quantum_sim.backends.numpy_backend import NumpyBackend
from quantum_sim.utils.expectation_value import ExpectationValueCalculator
from quantum_sim.gates.qaoa_cost_layer import QAOACostLayer
from quantum_sim.gates.qaoa_mixer_layer import QAOAMixerLayer
from quantum_sim.gates.hadamard_block import HadamardBlock
from quantum_sim.core.register import Register


def create_qaoa_ansatz_for_mapper(graph: nx.Graph, p_layers: int) -> QuantumCircuit:
    """
    Helper to construct the QAOA ansatz circuit for Max-Cut for the SweetSpotMapper.
    """
    num_qubits = graph.number_of_nodes()
    qaoa_circuit = QuantumCircuit(num_qubits, name=f"QAOA_Ansatz_p{p_layers}")
    
    initial_register = Register(size=num_qubits)
    initial_hadamard_block = HadamardBlock(initial_register, name="InitialHadamard")
    qaoa_circuit.add_sub_circuit(initial_hadamard_block, qubit_map_for_sub_circuit={i: i for i in range(num_qubits)})

    for i in range(p_layers):
        gamma_param = Parameter(f"gamma_{i}")
        beta_param = Parameter(f"beta_{i}")
        
        circuit_register = Register(size=num_qubits)
        
        cost_layer = QAOACostLayer(graph, circuit_register, gamma_param, name=f"CostLayer_{i}")
        qaoa_circuit.add_sub_circuit(
            cost_layer, 
            qubit_map_for_sub_circuit={j: j for j in range(num_qubits)},
            param_prefix=f"layer{i}_cost"
        )
        
        mixer_layer = QAOAMixerLayer(circuit_register, beta_param, name=f"MixerLayer_{i}")
        qaoa_circuit.add_sub_circuit(
            mixer_layer, 
            qubit_map_for_sub_circuit={j: j for j in range(num_qubits)},
            param_prefix=f"layer{i}_mixer"
        )

    return qaoa_circuit


class SweetSpotMapper:
    """
    Systematically maps the optimal QAOA cost as a function of circuit depth (p)
    under specified noise conditions to find the "Sweet Spot".
    """

    def __init__(self, graph: nx.Graph, num_qubits: int,
                 t1_times: Dict[int, float], t2_times: Dict[int, float], p_ex: float = 0.0,
                 depolarizing_noise_prob: float = 0.0):
        self.graph = graph
        self.num_qubits = num_qubits
        self.t1_times = t1_times
        self.t2_times = t2_times
        self.p_ex = p_ex
        self.depolarizing_noise_prob = depolarizing_noise_prob
        self.results: Dict[int, float] = {}
        self.optimal_params_history: Dict[int, np.ndarray] = {}

    def _setup_noisy_backend(self) -> NumpyBackend:
        per_qubit_noise_channels = {}
        if self.depolarizing_noise_prob > 0:
            for q_id in range(self.num_qubits):
                per_qubit_noise_channels[q_id] = [DepolarizingChannel(self.depolarizing_noise_prob)]
        
        backend = NumpyBackend(
            num_qubits=self.num_qubits,
            t1_times=self.t1_times,
            t2_times=self.t2_times,
            p_ex=self.p_ex,
            per_qubit_noise_channels=per_qubit_noise_channels
        )
        return backend

    def map_sweet_spot(self, max_p_layers: int = 6, optimizer_maxiter: int = 100) -> Dict[int, float]:
        exp_val_calculator = ExpectationValueCalculator(self.num_qubits)
        
        for p in range(1, max_p_layers + 1):
            backend = self._setup_noisy_backend()
            ansatz = create_qaoa_ansatz_for_mapper(self.graph, p)
            initial_params = np.random.uniform(0, 2*np.pi, size=len(ansatz.get_parameters()))

            optimizer = QAOAOptimizer(ansatz, backend, self.graph, exp_val_calculator,
                                      method='COBYLA', maxiter=optimizer_maxiter)
            
            min_energy, opt_params = optimizer.optimize(initial_params)
            self.results[p] = min_energy
            self.optimal_params_history[p] = opt_params
            
        return self.results

    def plot_sweet_spot(self, filename: str = "qaoa_sweet_spot.png", title_suffix: str = ""):
        if not self.results:
            return

        p_values = sorted(self.results.keys())
        min_energies = [self.results[p] for p in p_values]

        plt.figure(figsize=(10, 6))
        plt.plot(p_values, min_energies, marker='o', linestyle='-', color='indigo')
        plt.title(f"QAOA Min Energy vs. Depth {title_suffix}", fontsize=14)
        plt.xlabel("Circuit Depth (p)", fontsize=12)
        plt.ylabel("Expectation Value (Lower is Better)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(p_values)
        
        if min_energies:
            best_p_idx = np.argmin(min_energies)
            best_p_val = p_values[best_p_idx]
            plt.axvline(x=best_p_val, color='red', linestyle='--', label=f'Sweet Spot at p={best_p_val}')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(filename)
