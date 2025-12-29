import numpy as np
import networkx as nx
from typing import Dict
from quantum_sim.core.register import Register
from quantum_sim.core.parameter import Parameter
from quantum_sim.core.circuit import QuantumCircuit
from quantum_sim.gates.qaoa_cost_layer import QAOACostLayer
from quantum_sim.gates.qaoa_mixer_layer import QAOAMixerLayer
from quantum_sim.backend.numpy_backend import NumpyBackend
from quantum_sim.optimizer.qaoa_optimizer import QAOAOptimizer
from quantum_sim.utils.expectation_value import ExpectationValueCalculator


class SweetSpotMapper:
    """
    Maps the performance of QAOA as a function of depth p under noise.
    Identifies the 'sweet spot' where algorithmic gains meet noise decay.
    """

    def __init__(self, graph: nx.Graph, num_qubits: int, t1_times: Dict[int, float],
                 t2_times: Dict[int, float], p_ex: float, depolarizing_noise_prob: float):
        self.graph = graph
        self.num_qubits = num_qubits
        self.backend = NumpyBackend(
            num_qubits=num_qubits,
            t1_times=t1_times,
            t2_times=t2_times,
            p_ex=p_ex,
            depolarizing_noise_prob=depolarizing_noise_prob
        )
        self.calculator = ExpectationValueCalculator(num_qubits)

    def _create_ansatz(self, p: int) -> QuantumCircuit:
        register = Register(self.num_qubits)
        ansatz = QuantumCircuit(self.num_qubits, name=f"QAOA_p{p}")

        for layer_idx in range(p):
            gamma = Parameter(f"gamma_{layer_idx}")
            beta = Parameter(f"beta_{layer_idx}")

            cost_layer = QAOACostLayer(self.graph, register, gamma)
            mixer_layer = QAOAMixerLayer(register, beta)

            ansatz.add_gate(cost_layer, list(range(self.num_qubits)))
            ansatz.add_gate(mixer_layer, list(range(self.num_qubits)))

        return ansatz

    def map_sweet_spot(self, max_p_layers: int, optimizer_maxiter: int) -> Dict[int, float]:
        p_results = {}

        for p in range(1, max_p_layers + 1):
            ansatz = self._create_ansatz(p)
            num_params = len(ansatz.get_parameters())
            initial_guess = np.random.uniform(0, np.pi, num_params)

            optimizer = QAOAOptimizer(
                ansatz=ansatz,
                backend=self.backend,
                graph=self.graph,
                cost_op_calculator=self.calculator,
                maxiter=optimizer_maxiter
            )

            min_energy, _ = optimizer.optimize(initial_guess)
            p_results[p] = min_energy
            print(f"  Finished p={p}: Min Energy = {min_energy:.4f}")

        return p_results
