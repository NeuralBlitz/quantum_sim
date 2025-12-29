# quantum_sim/gates/qaoa_cost_layer.py

from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np
import networkx as nx

from quantum_sim.core.circuit import CircuitComponent, QuantumCircuit
from quantum_sim.core.register import Register
from quantum_sim.core.parameter import Parameter
from quantum_sim.gates.two_qubit_gates import CNOT
from quantum_sim.gates.parametric_gates import RZ

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit


class QAOACostLayer(CircuitComponent):
    """
    Implements the Cost Hamiltonian evolution layer for QAOA for a given graph.
    Applies e^(-i * gamma * H_C) where H_C = sum_edges (I - Z_u Z_v)/2.
    Each ZZ term expands to a CNOT, RZ(2*gamma), CNOT sequence.
    """

    def __init__(self, graph: nx.Graph, register: Register, gamma_param: Parameter, name: str = "QAOA_CostLayer"):
        if not isinstance(graph, nx.Graph):
            raise TypeError("Graph must be a NetworkX Graph instance.")
        if not isinstance(register, Register):
            raise TypeError("Register must be a Register instance.")
        if not isinstance(gamma_param, Parameter):
            raise TypeError("gamma_param must be a Parameter instance.")

        self.graph = graph
        self.register = register
        self.gamma_param = gamma_param
        self.name = name
        self.num_qubits = len(register)

        self._internal_circuit = QuantumCircuit(self.num_qubits, name=f"{self.name}_Internal")
        self._build_layer()

        # Aggregate duration
        cnot_dur = CNOT().duration
        rz_dur = RZ(Parameter("dummy")).duration
        edge_count = len(self.graph.edges())
        self.duration = edge_count * (2 * cnot_dur + rz_dur) if edge_count else 0.0

    def _build_layer(self):
        """Constructs the sequence of gates for the cost layer."""
        cnot_gate = CNOT()
        for u, v in self.graph.edges():
            if u >= self.num_qubits or v >= self.num_qubits:
                raise ValueError(f"Graph edge ({u},{v}) outside register range {self.num_qubits}.")

            # Unique parameter for each RZ gate to allow the 2*gamma binding
            rz_p_name = f"{self.gamma_param.name}_edge_{u}{v}_RZ_angle"
            rz_gate_instance = RZ(angle_param=Parameter(rz_p_name))

            self._internal_circuit.add_gate(cnot_gate, target_qubit_local_ids=[u, v])
            self._internal_circuit.add_gate(rz_gate_instance, target_qubit_local_ids=[v])
            self._internal_circuit.add_gate(cnot_gate, target_qubit_local_ids=[u, v])

    def get_qiskit_circuit_instructions(self, qiskit_qc: "QiskitQuantumCircuit", qubit_map: Dict[int, int]):
        self._internal_circuit.get_qiskit_circuit_instructions(qiskit_qc, qubit_map)

    def apply_to_density_matrix(self, rho_tensor: np.ndarray, num_total_qubits: int,
                                qubit_map: Dict[int, int]) -> np.ndarray:
        return self._internal_circuit.apply_to_density_matrix(rho_tensor, num_total_qubits, qubit_map)

    def get_involved_qubit_local_ids(self) -> List[int]:
        return list(range(self.num_qubits))

    def get_visualization_info(self, offset_x: float, offset_y: float,
                               qubit_y_coords: Dict[int, float]) -> List[Dict[str, Any]]:
        info = self._internal_circuit.get_visualization_info(offset_x, offset_y, qubit_y_coords)
        if info:
            x_vals = [i.get('x', i.get('x_start', 0)) for i in info]
            x_end_vals = [i.get('x', i.get('x_end', 0)) for i in info]
            y_vals = [i.get('y', i.get('y_min', 0)) for i in info]
            y_max_vals = [i.get('y', i.get('y_max', 0)) for i in info]

            info.append({
                'type': 'layer_box',
                'name': self.name,
                'x_start': min(x_vals) - 0.2,
                'x_end': max(x_end_vals) + 0.2,
                'y_min': min(y_vals) - 0.2,
                'y_max': max(y_max_vals) + 0.2,
                'offset_x': offset_x
            })
        return info

    def get_display_name(self) -> str:
        return self.name

    def get_parameters(self) -> Dict[str, Parameter]:
        all_params = self._internal_circuit.get_parameters()
        all_params[self.gamma_param.name] = self.gamma_param
        return all_params

    def bind_parameters(self, parameter_bindings: Dict[str, float]):
        if self.gamma_param.name in parameter_bindings:
            self.gamma_param.bind(parameter_bindings[self.gamma_param.name])
            gamma_val = self.gamma_param.get_value()

            # Update the RZ gates in the internal circuit with 2*gamma
            internal_params = self._internal_circuit.get_parameters()
            for u, v in self.graph.edges():
                rz_param_name = f"{self.gamma_param.name}_edge_{u}{v}_RZ_angle"
                if rz_param_name in internal_params:
                    internal_params[rz_param_name].bind(gamma_val * 2)

        self._internal_circuit.bind_parameters(parameter_bindings)

    def __repr__(self) -> str:
        g_val = f"{self.gamma_param.get_value():.3f}" if self.gamma_param.is_bound() else "unbound"
        return f"QAOACostLayer('{self.name}', {self.num_qubits} qubits, gamma={g_val}, [{self.duration*1e9:.0f}ns])"
