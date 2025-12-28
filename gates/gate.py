# quantum_sim/gates/gate.py

from abc import ABC, abstractmethod
from typing import List, Any, Dict, Union
import numpy as np
from quantum_sim.core.parameter import Parameter

class Gate(ABC):
    """
    Abstract Base Class for all quantum gates.
    Defines the common interface that all gates must implement.
    Now includes a 'duration' attribute for time-dependent noise.
    """
    def __init__(self, name: str, num_qubits: int, params: Dict[str, Parameter] = None, duration: float = 0.0):
        if not isinstance(name, str) or not name:
            raise ValueError("Gate name must be a non-empty string.")
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Gate must operate on at least one qubit.")
        if not isinstance(duration, (int, float)) or duration < 0:
            raise ValueError("Gate duration must be a non-negative number.")

        self.name = name
        self.num_qubits = num_qubits
        self.params: Dict[str, Parameter] = params if params is not None else {}
        self.duration = float(duration)

        if not all(isinstance(p, Parameter) for p in self.params.values()):
            raise TypeError("All values in 'params' dictionary must be instances of Parameter.")

    @abstractmethod
    def to_unitary(self) -> np.ndarray:
        """
        Returns the unitary matrix representation of the gate.
        If the gate is parametric, this method should ensure all required parameters
        are bound before constructing the matrix.
        """
        pass

    @abstractmethod
    def to_qiskit_instruction(self) -> Any:
        """
        Returns the corresponding Qiskit QuantumCircuit instruction.
        If the gate is parametric, this method should ensure all required parameters
        are bound before constructing the instruction.
        """
        pass
        
    def get_parameters(self) -> Dict[str, Parameter]:
        """Returns a dictionary of symbolic parameters associated with the gate."""
        return self.params

    def __repr__(self) -> str:
        param_str = ", ".join(f"{name}={param.get_value():.3f}" if param.is_bound() else f"{name}=unbound" for name, param in self.params.items())
        parts = [f"{self.name}Gate({self.num_qubits} qubits)"]
        if param_str:
            parts.append(f"({param_str})")
        if self.duration > 0:
            parts.append(f"[{self.duration*1e9:.0f}ns]")
        return "".join(parts)
