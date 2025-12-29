# quantum_sim/gates/parametric_gates.py

import numpy as np
from typing import Any
from qiskit.circuit.library import RXGate, RZGate

from quantum_sim.gates.gate import Gate
from quantum_sim.core.parameter import Parameter


class ParametricGate(Gate):
    """Base class for gates that operate on a single qubit and are parametric."""

    def __init__(self, name: str, angle_param: Parameter, duration: float):
        super().__init__(name, num_qubits=1, params={'angle': angle_param}, duration=duration)

    def _get_angle_value(self) -> float:
        """Helper to retrieve the numerical value of the bound angle."""
        return self.params['angle'].get_value()


class RX(ParametricGate):
    """Rotation around X-axis gate."""

    def __init__(self, angle_param: Parameter):
        # Hardware-realistic duration: 50 ns
        super().__init__("RX", angle_param, duration=50e-9)

    def to_unitary(self) -> np.ndarray:
        angle = self._get_angle_value()
        c = np.cos(angle / 2)
        s = np.sin(angle / 2)
        return np.array([[c, -1j * s],
                         [-1j * s, c]], dtype=complex)

    def to_qiskit_instruction(self) -> Any:
        angle = self._get_angle_value()
        return RXGate(angle)


class RZ(ParametricGate):
    """Rotation around Z-axis gate. Typically a virtual gate with 0 duration."""

    def __init__(self, angle_param: Parameter):
        # Virtual RZ gates usually have 0 duration on IBM-style backends
        super().__init__("RZ", angle_param, duration=0.0)

    def to_unitary(self) -> np.ndarray:
        angle = self._get_angle_value()
        return np.array([[np.exp(-1j * angle / 2), 0],
                         [0, np.exp(1j * angle / 2)]], dtype=complex)

    def to_qiskit_instruction(self) -> Any:
        angle = self._get_angle_value()
        return RZGate(angle)
