import numpy as np
from typing import List, Tuple, Dict, Any


class QuantumCircuit:
    """Represents a quantum circuit as a sequence of operations."""

    def __init__(self, num_qubits: int, name: str = "Circuit"):
        self.num_qubits = num_qubits
        self.name = name
        self.operations: List[Tuple[Any, List[int]]] = []

    def add_gate(self, gate, target_qubit_local_ids):
        """Adds a gate to the circuit."""
        if not isinstance(target_qubit_local_ids, list):
            target_qubit_local_ids = [target_qubit_local_ids]
        self.operations.append((gate, target_qubit_local_ids))

    def get_parameters(self) -> Dict[str, Any]:
        """Aggregates all parameters from gates in the circuit."""
        params = {}
        for gate, _ in self.operations:
            params.update(gate.get_parameters())
        return params

    def bind_parameters(self, bindings: Dict[str, float]):
        """Binds numerical values to the circuit parameters."""
        for gate, _ in self.operations:
            gate.bind_parameters(bindings)

    def get_visualization_info(self, offset_x: float, offset_y: float,
                               qubit_y_coords: Dict[int, float]) -> List[Dict[str, Any]]:
        """Returns a list of dictionaries for the CircuitDrawer."""
        info = []
        current_x = offset_x
        for gate, targets in self.operations:
            gate_info = {
                'type': 'gate',
                'name': gate.name,
                'num_qubits': len(targets),
                'x': current_x,
                'params': {k: p.get_value() for k, p in gate.params.items() if p.is_bound()}
            }
            if len(targets) == 1:
                gate_info['y'] = qubit_y_coords[targets[0]]
            elif len(targets) == 2:
                gate_info['control_y'] = qubit_y_coords[targets[0]]
                gate_info['target_y'] = qubit_y_coords[targets[1]]
            info.append(gate_info)
            current_x += 0.8
        return info
