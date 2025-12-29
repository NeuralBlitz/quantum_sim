from abc import ABC, abstractmethod
from typing import List, Dict, Any, TYPE_CHECKING
import numpy as np

from quantum_sim.core.parameter import Parameter
from quantum_sim.utils.jit_ops import jit_apply_unitary_to_rho

if TYPE_CHECKING:
    from quantum_sim.gates.gate import Gate
    from qiskit.circuit import QuantumCircuit as QiskitQuantumCircuit


class CircuitComponent(ABC):
    @abstractmethod
    def get_qiskit_circuit_instructions(self, qiskit_qc: "QiskitQuantumCircuit", qubit_map: Dict[int, int]): pass
    @abstractmethod
    def apply_to_density_matrix(self, rho: np.ndarray, n: int, q_map: Dict[int, int]) -> np.ndarray: pass
    @abstractmethod
    def get_involved_qubit_local_ids(self) -> List[int]: pass
    @abstractmethod
    def get_visualization_info(self, ox: float, oy: float, y_map: Dict[int, float]) -> List[Dict[str, Any]]: pass
    @abstractmethod
    def get_display_name(self) -> str: pass
    @abstractmethod
    def get_parameters(self) -> Dict[str, Parameter]: pass
    @abstractmethod
    def bind_parameters(self, bindings: Dict[str, float]): pass


class GateOperation(CircuitComponent):
    def __init__(self, gate: "Gate", target_qubit_local_ids: List[int]):
        self.gate = gate
        self.target_qubit_local_ids = target_qubit_local_ids

    def get_qiskit_circuit_instructions(self, qiskit_qc, qubit_map):
        indices = [qubit_map[qid] for qid in self.target_qubit_local_ids]
        qiskit_qc.append(self.gate.to_qiskit_instruction(), indices)

    def apply_to_density_matrix(self, rho_tensor, num_total_qubits, qubit_map):
        mapped_targets = np.array([qubit_map[qid] for qid in self.target_qubit_local_ids], dtype=np.int32)
        u = self.gate.to_unitary()
        u_dag = np.conj(u.T)
        return jit_apply_unitary_to_rho(rho_tensor, u, mapped_targets, num_total_qubits, u_dag)

    def get_involved_qubit_local_ids(self): return self.target_qubit_local_ids
    def get_display_name(self): return self.gate.name
    def get_parameters(self): return self.gate.get_parameters()
    def bind_parameters(self, bindings):
        for name, p_obj in self.gate.get_parameters().items():
            if p_obj.name in bindings: p_obj.bind(bindings[p_obj.name])
    def get_visualization_info(self, ox, oy, y_map): return [] # Simplified


class QuantumCircuit(CircuitComponent):
    def __init__(self, num_qubits: int, name: str = "MainCircuit"):
        self.num_qubits = num_qubits
        self.name = name
        self._components: List[CircuitComponent] = []

    def add_gate(self, gate, target_ids):
        op = GateOperation(gate, target_ids)
        self._components.append(op)
        return op

    def add_sub_circuit(self, sub_circuit, qubit_map, param_prefix=None):
        if param_prefix:
            param_obj_map = {}
            for name, obj in sub_circuit.get_parameters().items():
                new_obj = Parameter(f"{param_prefix}_{obj.name}")
                if obj.is_bound(): new_obj.bind(obj.get_value())
                param_obj_map[name] = new_obj
            sub_copy = self._deep_copy_and_reassign_parameters(sub_circuit, param_obj_map)
            mapped = _MappedSubCircuit(sub_copy, qubit_map)
        else:
            mapped = _MappedSubCircuit(sub_circuit, qubit_map)
        self._components.append(mapped)
        return mapped

    def _deep_copy_and_reassign_parameters(self, circuit, param_obj_map):
        new_circ = QuantumCircuit(circuit.num_qubits, name=f"{circuit.name}_copy")
        for comp in circuit._components:
            if isinstance(comp, GateOperation):
                gate = comp.gate
                params = gate.get_parameters()
                if params and 'angle' in params and params['angle'].name in param_obj_map:
                    new_gate = gate.__class__(angle_param=param_obj_map[params['angle'].name])
                    new_gate.duration = gate.duration
                else:
                    new_gate = gate
                new_circ.add_gate(new_gate, comp.target_qubit_local_ids)
            elif isinstance(comp, _MappedSubCircuit):
                new_sub = self._deep_copy_and_reassign_parameters(comp.sub_circuit, param_obj_map)
                new_circ.add_sub_circuit(new_sub, comp.qubit_map)
        return new_circ

    def get_parameters(self):
        all_p = {}
        for c in self._components:
            for p in c.get_parameters().values(): all_p[p.name] = p
        return all_p

    def bind_parameters(self, bindings):
        for c in self._components: c.bind_parameters(bindings)

    def apply_to_density_matrix(self, rho, num_total, qubit_map):
        current_rho = rho
        for c in self._components:
            current_rho = c.apply_to_density_matrix(current_rho, num_total, qubit_map)
        return current_rho

    def get_involved_qubit_local_ids(self): return []
    def get_visualization_info(self, ox, oy, y_map): return []
    def get_display_name(self): return self.name


class _MappedSubCircuit(CircuitComponent):
    def __init__(self, sub_circuit, qubit_map):
        self.sub_circuit = sub_circuit
        self.qubit_map = qubit_map

    def apply_to_density_matrix(self, rho, num_total, parent_map):
        sub_map = {sid: parent_map[pid] for sid, pid in self.qubit_map.items()}
        return self.sub_circuit.apply_to_density_matrix(rho, num_total, sub_map)

    def get_qiskit_circuit_instructions(self, qiskit_qc, parent_map):
        sub_map = {sid: parent_map[pid] for sid, pid in self.qubit_map.items()}
        self.sub_circuit.get_qiskit_circuit_instructions(qiskit_qc, sub_map)

    def get_involved_qubit_local_ids(self): return list(self.qubit_map.values())
    def get_display_name(self): return f"Sub:{self.sub_circuit.name}"
    def get_parameters(self): return self.sub_circuit.get_parameters()
    def bind_parameters(self, bindings): self.sub_circuit.bind_parameters(bindings)
    def get_visualization_info(self, ox, oy, y_map): return []
