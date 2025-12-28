# quantum_sim/gates/qaoa_mixer_layer.py

from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np

from quantum_sim.core.circuit import CircuitComponent, GateOperation, QuantumCircuit
from quantum_sim.core.register import Register
from quantum_sim.core.parameter import Parameter
from quantum_sim.gates.parametric_gates import RX # Parametric RX gate

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit

class QAOAMixerLayer(CircuitComponent):
    """
    Implements the Mixer Hamiltonian evolution layer for QAOA.
    Applies e^(-i * beta * H_M) where H_M = sum_i X_i.
    This expands to RX(2*beta) for each qubit.
    """
    def __init__(self, register: Register, beta_param: Parameter, name: str = "QAOA_MixerLayer"):
        if not isinstance(register, Register):
            raise TypeError("Register must be a Register instance.")
        if not isinstance(beta_param, Parameter):
            raise TypeError("beta_param must be a Parameter instance.")
        
        self.register = register
        self.beta_param = beta_param
        self.name = name
        self.num_qubits = len(register)
        
        self._internal_circuit = QuantumCircuit(self.num_qubits, name=f"{self.name}_Internal")
        self._build_layer()
        
        # Aggregate duration for the layer: All RX gates are executed in parallel conceptually,
        # so the layer duration is the duration of a single RX gate.
        rx_duration = RX(Parameter("dummy_for_duration")).duration
        self.duration = rx_duration


    def _build_layer(self):
        """Constructs the sequence of gates for the mixer layer."""
        for local_qubit_id in range(self.num_qubits):
            rx_angle_param_instance = Parameter(f"{self.beta_param.name}_qubit_{local_qubit_id}_RX_angle")
            rx_gate_instance = RX(angle_param=rx_angle_param_instance)
            self._internal_circuit.add_gate(rx_gate_instance, target_qubit_local_ids=[local_qubit_id])


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
        all_params[self.beta_param.name] = self.beta_param
        return all_params
    
    def bind_parameters(self, parameter_bindings: Dict[str, float]):
        self.beta_param.bind(parameter_bindings[self.beta_param.name])
        
        for local_qubit_id in range(self.num_qubits):
            rx_param_name = f"{self.beta_param.name}_qubit_{local_qubit_id}_RX_angle"
            if rx_param_name in self._internal_circuit.get_parameters():
                self._internal_circuit.get_parameters()[rx_param_name].bind(self.beta_param.get_value() * 2)

        self._internal_circuit.bind_parameters(parameter_bindings)

    def __repr__(self) -> str:
        beta_str = f"beta={self.beta_param.get_value():.3f}" if self.beta_param.is_bound() else "beta=unbound"
        return f"QAOAMixerLayer('{self.name}', {self.num_qubits} qubits, {beta_str}, duration=[{self.duration*1e9:.0f}ns])"
