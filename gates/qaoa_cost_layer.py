# quantum_sim/gates/qaoa_cost_layer.py

from typing import List, Tuple, Dict, Any, TYPE_CHECKING
import numpy as np
import networkx as nx

from quantum_sim.core.circuit import CircuitComponent, GateOperation, QuantumCircuit
from quantum_sim.core.register import Register
from quantum_sim.core.parameter import Parameter
from quantum_sim.gates.two_qubit_gates import CNOT
from quantum_sim.gates.parametric_gates import RZ # Parametric RZ gate

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit

class QAOACostLayer(CircuitComponent):
    """
    Implements the Cost Hamiltonian evolution layer for QAOA for a given graph.
    Applies e^(-i * gamma * H_C) where H_C = sum_edges (I - Z_u Z_v)/2.
    Each e^(-i * gamma * Z_u Z_v) term expands to a CNOT, RZ(2*gamma), CNOT sequence.
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
        
        # Aggregate duration for the layer: sum of durations of all gates in sequence
        # Assuming sequential execution for the CNOT-RZ-CNOT sequence for each edge
        cnot_duration = CNOT().duration
        rz_duration = RZ(Parameter("dummy_for_duration")).duration # RZ is typically 0 duration
        
        # Total duration for one edge term is 2 * CNOT_duration + RZ_duration
        # If edge terms are applied in parallel conceptually, duration is max over edges.
        # But for simulation, we apply them sequentially.
        self.duration = len(self.graph.edges()) * (2 * cnot_duration + rz_duration) if self.graph.edges() else 0.0


    def _build_layer(self):
        """Constructs the sequence of gates for the cost layer."""
        cnot_gate = CNOT()
        
        for u, v in self.graph.edges():
            if u >= self.num_qubits or v >= self.num_qubits:
                raise ValueError(f"Graph edge ({u},{v}) involves qubits outside register size {self.num_qubits}.")
            
            # Create a unique parameter instance for each RZ gate, then bind its value
            rz_angle_param_for_edge = Parameter(f"{self.gamma_param.name}_edge_{u}{v}_RZ_angle")
            rz_gate_instance = RZ(angle_param=rz_angle_param_for_edge)
            
            self._internal_circuit.add_gate(cnot_gate, target_qubit_local_ids=[u, v])
            self._internal_circuit.add_gate(rz_gate_instance, target_qubit_local_ids=[v]) 
            self._internal_circuit.add_gate(cnot_gate, target_qubit_local_ids=[u, v])


    def get_qiskit_circuit_instructions(self, qiskit_qc: "QiskitQuantumCircuit",
                                        qubit_map: Dict[int, int]):
        self._internal_circuit.get_qiskit_circuit_instructions(qiskit_qc, qubit_map)

    def apply_to_density_matrix(self, rho_tensor: np.ndarray, num_total_qubits: int,
                              qubit_map: Dict[int, int]) -> np.ndarray:
        return self._internal_circuit.apply_to_density_matrix(rho_tensor, num_total_qubits, qubit_map)

    def get_involved_qubit_local_ids(self) -> List[int]:
        return self._internal_circuit.get_involved_qubit_local_ids()

    def get_visualization_info(self, offset_x: float, offset_y: float,
                               qubit_y_coords: Dict[int, float]) -> List[Dict[str, Any]]:
        info = self._internal_circuit.get_visualization_info(offset_x, offset_y, qubit_y_coords)
        
        if info:
            min_y_comp = min(item.get('y', item.get('y_min', 0)) for item in info if 'y' in item or 'y_min' in item)
            max_y_comp = max(item.get('y', item.get('y_max', 0)) for item in info if 'y' in item or 'y_max' in item)
            min_x_comp = min(item.get('x', item.get('x_start', 0)) for item in info if 'x' in item or 'x_start' in item)
            max_x_comp = max(item.get('x', item.get('x_end', 0)) for item in info if 'x' in item or 'x_end' in item)
            
            info.append({
                'type': 'layer_box',
                'name': self.name,
                'x_start': min_x_comp - 0.2,
                'x_end': max_x_comp + 0.2,
                'y_min': min_y_comp - 0.2,
                'y_max': max_y_comp + 0.2,
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
        self.gamma_param.bind(parameter_bindings[self.gamma_param.name])
        
        # Update the RZ gates in the internal circuit with the *bound* gamma value (2*gamma)
        for u, v in self.graph.edges():
            rz_param_name = f"{self.gamma_param.name}_edge_{u}{v}_RZ_angle"
            if rz_param_name in self._internal_circuit.get_parameters():
                self._internal_circuit.get_parameters()[rz_param_name].bind(self.gamma_param.get_value() * 2)
        
        self._internal_circuit.bind_parameters(parameter_bindings)


    def __repr__(self) -> str:
        gamma_str = f"gamma={self.gamma_param.get_value():.3f}" if self.gamma_param.is_bound() else "gamma=unbound"
        return f"QAOACostLayer('{self.name}', {self.num_qubits} qubits, {gamma_str}, duration=[{self.duration*1e9:.0f}ns])"
