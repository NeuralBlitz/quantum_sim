from typing import List, Any, Dict, Optional, Union
import numpy as np

class CircuitComponent:
    """Base class for anything that can be added to a quantum circuit."""
    
    def apply_to_density_matrix(self, rho: np.ndarray, num_qubits: int, qubit_map: Dict[int, int]) -> np.ndarray:
        """Must be implemented by subclasses to evolve the quantum state."""
        raise NotImplementedError("Subclasses must implement apply_to_density_matrix")

    def get_involved_qubit_local_ids(self) -> List[int]:
        """Returns the list of qubits this component acts upon."""
        raise NotImplementedError

class GateOperation(CircuitComponent):
    """Represents a specific gate acting on specific qubits."""
    
    def __init__(self, gate: Any, qubit_ids: List[int]):
        self.gate = gate
        self.qubit_ids = qubit_ids

    def get_involved_qubit_local_ids(self) -> List[int]:
        return self.qubit_ids

    def apply_to_density_matrix(self, rho: np.ndarray, num_qubits: int, qubit_map: Dict[int, int]) -> np.ndarray:
        # Map local circuit IDs to global backend IDs
        global_targets = [qubit_map[qid] for qid in self.qubit_ids]
        return self.gate.apply_to_density_matrix(rho, global_targets, num_qubits)

class QuantumCircuit(CircuitComponent):
    """A collection of GateOperations and other CircuitComponents."""
    
    def __init__(self, num_qubits: int, name: str = "Circuit"):
        self.num_qubits = num_qubits
        self.name = name
        # Use List[CircuitComponent] to ensure mypy can track apply_to_density_matrix
        self._components: List[CircuitComponent] = []

    def add(self, component: CircuitComponent):
        """Adds a component (Gate or Sub-circuit) to the circuit."""
        self._components.append(component)

    def get_involved_qubit_local_ids(self) -> List[int]:
        """Returns all qubits involved in this circuit's components."""
        involved = set()
        for comp in self._components:
            involved.update(comp.get_involved_qubit_local_ids())
        return sorted(list(involved))

    def apply_to_density_matrix(self, rho: np.ndarray, num_qubits: int, qubit_map: Dict[int, int]) -> np.ndarray:
        """Evolves the state through all components in the circuit."""
        for component in self._components:
            rho = component.apply_to_density_matrix(rho, num_qubits, qubit_map)
        return rho

    def get_qiskit_circuit_instructions(self) -> List[Any]:
        """Stub to satisfy external backend requirements (like Qiskit)."""
        instructions: List[Any] = []
        for comp in self._components:
            if hasattr(comp, 'get_qiskit_circuit_instructions'):
                instructions.extend(comp.get_qiskit_circuit_instructions())
        return instructions
