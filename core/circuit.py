# quantum_sim/core/circuit.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING, Tuple
import numpy as np
import string

from quantum_sim.core.parameter import Parameter
from quantum_sim.utils.jit_ops import jit_apply_unitary_to_rho

if TYPE_CHECKING:
    from quantum_sim.gates.gate import Gate
    from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit


class CircuitComponent(ABC):
    """
    Abstract Base Class for all components that can be added to a QuantumCircuit.
    This is the core of the Composite Pattern.
    """
    @abstractmethod
    def get_qiskit_circuit_instructions(self, qiskit_qc: "QiskitQuantumCircuit",
                                        qubit_map: Dict[int, int]):
        """
        Applies this component's operations to a QiskitQuantumCircuit instance,
        using the provided qubit_map to translate local qubit IDs.
        """
        pass

    @abstractmethod
    def apply_to_density_matrix(self, rho_tensor: np.ndarray, num_total_qubits: int,
                              qubit_map: Dict[int, int]) -> np.ndarray:
        """
        Applies this component's operations to a raw NumPy density matrix tensor (rho)
        using np.einsum: rho -> U rho U^dagger.
        rho_tensor: (2,2,...,2) for row indices, (2,2,...,2) for col indices. Total 2N dims.
        """
        pass

    @abstractmethod
    def get_involved_qubit_local_ids(self) -> List[int]:
        """
        Returns a list of local qubit IDs (0-indexed) that this component
        directly involves within its own scope.
        """
        pass

    @abstractmethod
    def get_visualization_info(self, offset_x: float, offset_y: float,
                               qubit_y_coords: Dict[int, float]) -> List[Dict[str, Any]]:
        """
        Returns information for visualization (e.g., gate type, position, affected qubits).
        """
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Returns a string name for display purposes."""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Parameter]:
        """
        Recursively collects all unique symbolic parameters used by this component
        and its nested components.
        """
        pass
    
    @abstractmethod
    def bind_parameters(self, parameter_bindings: Dict[str, float]):
        """
        Recursively binds numerical values to symbolic parameters across this
        component and its nested components.
        """
        pass


class GateOperation(CircuitComponent):
    """
    A leaf node in the Composite Pattern. Represents a single Gate applied
    to specific qubits within a circuit. Now operates on density matrices
    using JIT-accelerated tensor contractions.
    """
    def __init__(self, gate: "Gate", target_qubit_local_ids: List[int]):
        if not isinstance(gate, Gate):
            raise TypeError("gate must be an instance of Gate.")
        if not isinstance(target_qubit_local_ids, list) or \
           not all(isinstance(q_id, int) for q_id in target_qubit_local_ids):
            raise ValueError("target_qubit_local_ids must be a list of integers.")
        if len(target_qubit_local_ids) != gate.num_qubits:
            raise ValueError(f"{gate.name} requires {gate.num_qubits} qubits, but {len(target_qubit_local_ids)} were provided.")

        self.gate = gate
        self.target_qubit_local_ids = target_qubit_local_ids

    def get_qiskit_circuit_instructions(self, qiskit_qc: "QiskitQuantumCircuit",
                                        qubit_map: Dict[int, int]):
        qiskit_target_indices = [qubit_map[local_id] for local_id in self.target_qubit_local_ids]
        
        if self.gate.num_qubits == 1:
            qiskit_qc.append(self.gate.to_qiskit_instruction(), [qiskit_target_indices[0]])
        elif self.gate.num_qubits == 2:
            qiskit_qc.append(self.gate.to_qiskit_instruction(), [qiskit_target_indices[0], qiskit_target_indices[1]])
        else:
            qiskit_qc.append(self.gate.to_qiskit_instruction(), qiskit_target_indices)

    def apply_to_density_matrix(self, rho_tensor: np.ndarray, num_total_qubits: int,
                              qubit_map: Dict[int, int]) -> np.ndarray:
        """
        Applies this component's unitary operation (U) to a raw NumPy density matrix tensor (rho)
        using JIT-accelerated np.einsum: rho -> U rho U^dagger.
        rho_tensor: (2,2,...,2) for row indices, (2,2,...,2) for col indices. Total 2N dims.
        """
        mapped_target_qubits = np.array([qubit_map[local_id] for local_id in self.target_qubit_local_ids], dtype=np.int32)
        
        if not all(0 <= qid < num_total_qubits for qid in mapped_target_qubits):
            raise ValueError(f"Mapped qubit IDs {mapped_target_qubits} out of bounds for {num_total_qubits} total qubits.")

        gate_unitary = self.gate.to_unitary()
        gate_unitary_dag = np.conj(gate_unitary.T)
        
        new_rho_tensor = jit_apply_unitary_to_rho(rho_tensor, gate_unitary, 
                                                  mapped_target_qubits, num_total_qubits, 
                                                  gate_unitary_dag)
        
        return new_rho_tensor

    def get_involved_qubit_local_ids(self) -> List[int]:
        return self.target_qubit_local_ids

    def get_visualization_info(self, offset_x: float, offset_y: float,
                               qubit_y_coords: Dict[int, float]) -> List[Dict[str, Any]]:
        info = []
        
        if self.gate.num_qubits == 1:
            target_q_id = self.target_qubit_local_ids[0]
            info.append({
                'type': 'gate',
                'name': self.gate.name,
                'num_qubits': self.gate.num_qubits,
                'x': offset_x,
                'y': qubit_y_coords[target_q_id],
                'target_qubit_id': target_q_id,
                'component_id': id(self),
                'params': {name: p.get_value() for name, p in self.gate.get_parameters().items() if p.is_bound()}
            })
        elif self.gate.num_qubits == 2:
            control_q_id = self.target_qubit_local_ids[0]
            target_q_id = self.target_qubit_local_ids[1]

            info.append({
                'type': 'gate',
                'name': self.gate.name,
                'num_qubits': self.gate.num_qubits,
                'x': offset_x,
                'control_y': qubit_y_coords[control_q_id],
                'target_y': qubit_y_coords[target_q_id],
                'control_qubit_id': control_q_id,
                'target_qubit_id': target_q_id,
                'component_id': id(self),
                'params': {name: p.get_value() for name, p in self.gate.get_parameters().items() if p.is_bound()}
            })
        else:
            min_y = min(qubit_y_coords[q_id] for q_id in self.target_qubit_local_ids)
            max_y = max(qubit_y_coords[q_id] for q_id in self.target_qubit_local_ids)
            
            info.append({
                'type': 'multi_qubit_gate_box',
                'name': self.gate.name,
                'num_qubits': self.gate.num_qubits,
                'x': offset_x,
                'y_min': min_y - 0.25,
                'y_max': max_y + 0.25,
                'involved_qubit_ids': self.target_qubit_local_ids,
                'component_id': id(self),
                'params': {name: p.get_value() for name, p in self.gate.get_parameters().items() if p.is_bound()}
            })

        return info
    
    def get_parameters(self) -> Dict[str, Parameter]:
        return self.gate.get_parameters()

    def bind_parameters(self, parameter_bindings: Dict[str, float]):
        for name, param_obj in self.gate.get_parameters().items():
            if param_obj.name in parameter_bindings:
                param_obj.bind(parameter_bindings[param_obj.name])

    def get_display_name(self) -> str:
        return self.gate.name


class QuantumCircuit(CircuitComponent):
    """
    A composite node in the Composite Pattern. Represents a quantum circuit
    that can contain other CircuitComponents (gates or sub-circuits).
    """
    def __init__(self, num_qubits: int, name: str = "MainCircuit"):
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Circuit must have a positive number of qubits.")
        self.num_qubits = num_qubits
        self.name = name
        self._components: List[CircuitComponent] = []

    def add_gate(self, gate: "Gate", target_qubit_local_ids: List[int]):
        """Adds a single gate operation to the circuit."""
        op = GateOperation(gate, target_qubit_local_ids)
        self._components.append(op)
        return op

    def add_sub_circuit(self, sub_circuit: 'QuantumCircuit',
                        qubit_map_for_sub_circuit: Dict[int, int],
                        param_prefix: str = None):
        """
        Adds another QuantumCircuit as a sub-circuit.
        qubit_map_for_sub_circuit: Maps local IDs of sub_circuit to local IDs of this circuit.
        param_prefix: Optional prefix for parameters within the sub-circuit to avoid collisions.
        """
        if not isinstance(sub_circuit, QuantumCircuit):
            raise TypeError("sub_circuit must be an instance of QuantumCircuit.")
        if len(qubit_map_for_sub_circuit) != sub_circuit.num_qubits:
            raise ValueError(f"Qubit map size ({len(qubit_map_for_sub_circuit)}) must match sub-circuit's qubit count ({sub_circuit.num_qubits}).")
        if not all(0 <= local_id < self.num_qubits for local_id in qubit_map_for_sub_circuit.values()):
            raise ValueError("Mapped qubits for sub-circuit are out of bounds for this circuit.")

        if param_prefix:
            param_obj_map = {} 
            original_params = sub_circuit.get_parameters()
            
            for original_param_name, original_param_obj in original_params.items():
                prefixed_name = f"{param_prefix}_{original_param_obj.name}"
                new_param_obj = Parameter(prefixed_name)
                if original_param_obj.is_bound():
                    new_param_obj.bind(original_param_obj.get_value())
                param_obj_map[original_param_name] = new_param_obj
            
            sub_circuit_copy = self._deep_copy_and_reassign_parameters(sub_circuit, param_obj_map)
            mapped_sub_circuit = _MappedSubCircuit(sub_circuit_copy, qubit_map_for_sub_circuit)
        else:
            mapped_sub_circuit = _MappedSubCircuit(sub_circuit, qubit_map_for_sub_circuit)

        self._components.append(mapped_sub_circuit)
        return mapped_sub_circuit

    def _deep_copy_and_reassign_parameters(self, circuit: 'QuantumCircuit', param_obj_map: Dict[str, Parameter]) -> 'QuantumCircuit':
        """
        Helper method to create a deep copy of a circuit and reassign its
        parameters based on a mapping of original parameter names to new Parameter objects.
        """
        new_circuit = QuantumCircuit(circuit.num_qubits, name=f"{circuit.name}_copy")
        for component in circuit._components:
            if isinstance(component, GateOperation):
                original_gate = component.gate
                if isinstance(original_gate, Gate) and original_gate.get_parameters():
                    if 'angle' in original_gate.get_parameters() and original_gate.get_parameters()['angle'].name in param_obj_map:
                         new_angle_param = param_obj_map[original_gate.get_parameters()['angle'].name]
                         new_gate = original_gate.__class__(angle_param=new_angle_param)
                         new_gate.duration = original_gate.duration 
                    else:
                         new_gate = original_gate
                else:
                    new_gate = original_gate 
                
                new_circuit.add_gate(new_gate, component.target_qubit_local_ids)

            elif isinstance(component, _MappedSubCircuit):
                new_sub_circuit = self._deep_copy_and_reassign_parameters(component.sub_circuit, param_obj_map)
                new_circuit.add_sub_circuit(new_sub_circuit, component.qubit_map, param_prefix=None)
            elif isinstance(component, QuantumCircuit):
                 new_sub_circuit = self._deep_copy_and_reassign_parameters(component, param_obj_map)
                 new_circuit.add_sub_circuit(new_sub_circuit, {i:i for i in range(component.num_qubits)}, param_prefix=None)
        return new_circuit


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

    def get_qiskit_circuit_instructions(self, qiskit_qc: "QiskitQuantumCircuit",
                                        qubit_map: Dict[int, int]):
        for param_obj in self.get_parameters().values():
            if not param_obj.is_bound():
                raise ValueError(f"Parameter '{param_obj.name}' is not bound before Qiskit instruction generation.")
        
        for component in self._components:
            component.get_qiskit_circuit_instructions(qiskit_qc, qubit_map)

    def apply_to_density_matrix(self, rho_tensor: np.ndarray, num_total_qubits: int,
                              qubit_map: Dict[int, int]) -> np.ndarray:
        for param_obj in self.get_parameters().values():
            if not param_obj.is_bound():
                raise ValueError(f"Parameter '{param_obj.name}' is not bound before density matrix simulation.")

        current_rho = rho_tensor
        for component in self._components:
            current_rho = component.apply_to_density_matrix(current_rho, num_total_qubits, qubit_map)
        return current_rho

    def get_involved_qubit_local_ids(self) -> List[int]:
        involved_ids = set()
        for component in self._components:
            involved_ids.update(component.get_involved_qubit_local_ids())
        return sorted(list(involved_ids))
    
    def get_visualization_info(self, offset_x: float, offset_y: float,
                               qubit_y_coords: Dict[int, float]) -> List[Dict[str, Any]]:
        info = []
        current_x = offset_x
        for component in self._components:
            component_info = component.get_visualization_info(current_x, offset_y, qubit_y_coords)
            info.extend(component_info)
            
            if component_info:
                max_x_component = max(item.get('x', item.get('x_end', current_x)) for item in component_info)
                current_x = max_x_component + 0.5 
            else:
                current_x += 0.5
        return info

    def get_display_name(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"QuantumCircuit('{self.name}', {self.num_qubits} qubits, {len(self._components)} components)"


class _MappedSubCircuit(CircuitComponent):
    """
    An internal wrapper for sub-circuits, handling qubit mapping
    when they are added to a parent circuit.
    """
    def __init__(self, sub_circuit: QuantumCircuit, qubit_map_for_sub_circuit: Dict[int, int]):
        self.sub_circuit = sub_circuit
        self.qubit_map = qubit_map_for_sub_circuit

    def get_qiskit_circuit_instructions(self, qiskit_qc: "QiskitQuantumCircuit",
                                        parent_qubit_map: Dict[int, int]):
        qiskit_sub_qubit_map = {
            sub_local_id: parent_qubit_map[parent_local_id]
            for sub_local_id, parent_local_id in self.qubit_map.items()
        }
        self.sub_circuit.get_qiskit_circuit_instructions(qiskit_qc, qiskit_sub_qubit_map)

    def apply_to_density_matrix(self, rho_tensor: np.ndarray, num_total_qubits: int,
                              parent_qubit_map: Dict[int, int]) -> np.ndarray:
        numpy_sub_qubit_map = {
            sub_local_id: parent_qubit_map[parent_local_id]
            for sub_local_id, parent_local_id in self.qubit_map.items()
        }
        return self.sub_circuit.apply_to_density_matrix(rho_tensor, num_total_qubits, numpy_sub_qubit_map)

    def get_involved_qubit_local_ids(self) -> List[int]:
        return list(self.qubit_map.values())

    def get_visualization_info(self, offset_x: float, offset_y: float,
                               qubit_y_coords: Dict[int, float]) -> List[Dict[str, Any]]:
        sub_circuit_info = self.sub_circuit.get_visualization_info(offset_x, offset_y, qubit_y_coords)
        
        adjusted_info = []
        for item in sub_circuit_info:
            if 'target_qubit_id' in item:
                original_sub_local_id = item['target_qubit_id']
                mapped_parent_local_id = self.qubit_map.get(original_sub_local_id)
                if mapped_parent_local_id is not None:
                    new_item = item.copy()
                    new_item['target_qubit_id'] = mapped_parent_local_id
                    new_item['y'] = qubit_y_coords[mapped_parent_local_id]
                    adjusted_info.append(new_item)
                else:
                    pass 
            elif 'control_qubit_id' in item and 'target_qubit_id' in item:
                 original_control_id = item['control_qubit_id']
                 original_target_id = item['target_qubit_id']
                 mapped_parent_control_id = self.qubit_map.get(original_control_id)
                 mapped_parent_target_id = self.qubit_map.get(original_target_id)
                 if mapped_parent_control_id is not None and mapped_parent_target_id is not None:
                     new_item = item.copy()
                     new_item['control_qubit_id'] = mapped_parent_control_id
                     new_item['target_qubit_id'] = mapped_parent_target_id
                     new_item['control_y'] = qubit_y_coords[mapped_parent_control_id]
                     new_item['target_y'] = qubit_y_coords[mapped_parent_target_id]
                     adjusted_info.append(new_item)
                 else:
                     pass
            elif 'involved_qubit_ids' in item:
                original_involved_ids = item['involved_qubit_ids']
                mapped_parent_involved_ids = [self.qubit_map.get(q_id) for q_id in original_involved_ids if self.qubit_map.get(q_id) is not None]
                if mapped_parent_involved_ids:
                    min_y = min(qubit_y_coords[q_id] for q_id in mapped_parent_involved_ids)
                    max_y = max(qubit_y_coords[q_id] for q_id in mapped_parent_involved_ids)
                    new_item = item.copy()
                    new_item['y_min'] = min_y - 0.25
                    new_item['y_max'] = max_y + 0.25
                    new_item['involved_qubit_ids'] = mapped_parent_involved_ids
                    adjusted_info.append(new_item)
                else:
                    pass
            else:
                adjusted_info.append(item) 
        
        if adjusted_info:
            min_y_comp = min(item.get('y', item.get('y_min', 0)) for item in adjusted_info if 'y' in item or 'y_min' in item)
            max_y_comp = max(item.get('y', item.get('y_max', 0)) for item in adjusted_info if 'y' in item or 'y_max' in item)
            min_x_comp = min(item.get('x', item.get('x_start', 0)) for item in adjusted_info if 'x' in item or 'x_start' in item)
            max_x_comp = max(item.get('x', item.get('x_end', 0)) for item in adjusted_info if 'x' in item or 'x_end' in item)
            
            adjusted_info.append({
                'type': 'sub_circuit_box',
                'name': self.sub_circuit.name,
                'x_start': min_x_comp - 0.2,
                'x_end': max_x_comp + 0.2,
                'y_min': min_y_comp - 0.2,
                'y_max': max_y_comp + 0.2,
                'offset_x': offset_x
            })

        return adjusted_info

    def get_parameters(self) -> Dict[str, Parameter]:
        return self.sub_circuit.get_parameters()
    
    def bind_parameters(self, parameter_bindings: Dict[str, float]):
        self.sub_circuit.bind_parameters(parameter_bindings)

    def get_display_name(self) -> str:
        return f"SubCircuit:{self.sub_circuit.name}"
