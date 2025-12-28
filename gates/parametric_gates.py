# quantum_sim/gates/parametric_gates.py

import numpy as np
from typing import Any, Dict
from qiskit.circuit.library import RXGate, RZGate

from quantum_sim.gates.gate import Gate
from quantum_sim.core.parameter import Parameter

class ParametricGate(Gate):
    """Base class for gates that operate on a single qubit and are parametric."""
    def __init__(self, name: str, angle_param: Parameter, duration: float):
        super().__init__(name, num_qubits=1, params={'angle': angle_param}, duration=duration)

    def _get_angle_value(self) -> float:
        return self.params['angle'].get_value()

class RX(ParametricGate):
    """Rotation around X-axis gate."""
    def __init__(self, angle_param: Parameter):
        super().__init__("RX", angle_param, duration=50e-9) # 50 ns duration

    def to_unitary(self) -> np.ndarray:
        angle = self._get_angle_value()
        return np.array([[np.cos(angle/2), -1j*np.sin(angle/2)],
                         [-1j*np.sin(angle/2), np.cos(angle/2)]])

    def to_qiskit_instruction(self) -> Any:
        angle = self._get_angle_value()
        return RXGate(angle)

class RZ(ParametricGate):
    """Rotation around Z-axis gate. Typically a virtual gate with 0 duration."""
    def __init__(self, angle_param: Parameter):
        super().__init__("RZ", angle_param, duration=0e-9) # 0 ns duration

    def to_unitary(self) -> np.ndarray:
        angle = self._get_angle_value()
        return np.array([[np.exp(-1j * angle/2), 0],
                         [0, np.exp(1j * angle/2)]])

    def to_qiskit_instruction(self) -> Any:
        angle = self._get_angle_value()
        return RZGate(angle)
