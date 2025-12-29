# quantum_sim/gates/single_qubit_gates.py

import numpy as np
from typing import Any
from qiskit.circuit.library import HGate, XGate

from quantum_sim.gates.gate import Gate


class SingleQubitGate(Gate):
    """Base class for gates that operate on a single qubit."""

    def __init__(self, name: str, unitary: np.ndarray, duration: float):
        if unitary.shape != (2, 2):
            raise ValueError("Unitary for a single-qubit gate must be 2x2.")
        super().__init__(name, num_qubits=1, duration=duration)
        self._unitary = unitary

    def to_unitary(self) -> np.ndarray:
        """Returns the fixed unitary matrix for this gate."""
        return self._unitary


class Hadamard(SingleQubitGate):
    """
    Hadamard gate applies superposition.
    Maps |0> to (|0> + |1>) / sqrt(2) and |1> to (|0> - |1>) / sqrt(2).
    """

    def __init__(self):
        val = 1 / np.sqrt(2)
        unitary = np.array([[val, val],
                            [val, -val]], dtype=complex)
        super().__init__("H", unitary, duration=50e-9)

    def to_qiskit_instruction(self) -> Any:
        return HGate()


class PauliX(SingleQubitGate):
    """
    Pauli-X (NOT) gate flips the qubit state.
    Maps |0> to |1> and |1> to |0>.
    """

    def __init__(self):
        unitary = np.array([[0, 1],
                            [1, 0]], dtype=complex)
        super().__init__("X", unitary, duration=50e-9)

    def to_qiskit_instruction(self) -> Any:
        return XGate()
