# quantum_sim/gates/hadamard_block.py

from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np

from quantum_sim.core.circuit import CircuitComponent, QuantumCircuit, GateOperation
from quantum_sim.core.register import Register
from quantum_sim.gates.single_qubit_gates import Hadamard
from quantum_sim.core.parameter import Parameter # For type hinting duration calculation

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit

class HadamardBlock(CircuitComponent):
    """
    A custom composite gate that applies a Hadamard gate to every qubit
    in a given quantum register. This demonstrates building a reusable
    block using the Composite Pattern.
    """
    def __init__(self, register: Register, name: str = "HadamardBlock"):
        if not isinstance(register, Register):
            raise TypeError("HadamardBlock must be initialized with a Register instance.")
        self.register = register
        self.name = name
        self.num_qubits = len(register)

        self._components: List[GateOperation] = []
        self._build_block()
        
        # Aggregate duration for the block: Max duration of any gate if applied in parallel,
        # or sum if sequential. For a block like this, gates are conceptually parallel.
        self.duration = max(comp.gate.duration for comp in self._components) if self._components else 0.0


    def _build_block(self):
        """Internal method to populate the block with Hadamard gates."""
        hadamard_gate = Hadamard()
        for local_qubit_id in range(self.num_qubits):
            self._components.append(GateOperation(hadamard_gate, [local_qubit_id]))

    def get_qiskit_circuit_instructions(self, qiskit_qc: "QiskitQuantumCircuit",
                                        qubit_map: Dict[int, int]):
        for component in self._components:
            component.get_qiskit_circuit_instructions(qiskit_qc, qubit_map)

    def apply_to_density_matrix(self, rho_tensor: np.ndarray, num_total_qubits: int,
                              qubit_map: Dict[int, int]) -> np.ndarray:
        current_rho = rho_tensor
        for component in self._components:
            current_rho = component.apply_to_density_matrix(current_rho, num_total_qubits, qubit_map)
        return current_rho

    def get_involved_qubit_local_ids(self) -> List[int]:
        return list(range(self.num_qubits))

    def get_visualization_info(self, offset_x: float, offset_y: float,
                               qubit_y_coords: Dict[int, float]) -> List[Dict[str, Any]]:
        info = []
        for component in self._components:
            gate_info = component.get_visualization_info(offset_x, offset_y, qubit_y_coords)
            info.extend(gate_info)
        
        if info:
            min_y_comp = min(item.get('y', item.get('y_min', 0)) for item in info if 'y' in item or 'y_min' in item)
            max_y_comp = max(item.get('y', item.get('y_max', 0)) for item in info if 'y' in item or 'y_max' in item)
            min_x_comp = min(item.get('x', item.get('x_start', 0)) for item in info if 'x' in item or 'x_start' in item)
            max_x_comp = max(item.get('x', item.get('x_end', 0)) for item in info if 'x' in item or 'x_end' in item)
            
            info.append({
                'type': 'block_box',
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
        all_params: Dict[str, Parameter] = {}
        for component in self._components:
            component_params = component.get_parameters()
            for name, param_obj in component_params.items():
                all_params[param_obj.name] = param_obj
        return all_params
    
    def bind_parameters(self, parameter_bindings: Dict[str, float]):
        for component in self._components:
            component.bind_parameters(parameter_bindings)

    def __repr__(self) -> str:
        return f"HadamardBlock('{self.name}', {self.num_qubits} qubits, duration=[{self.duration*1e9:.0f}ns])"
