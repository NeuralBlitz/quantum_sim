# quantum_sim/gates/two_qubit_gates.py

import numpy as np
from typing import Any
from qiskit.circuit.library import CXGate

from quantum_sim.gates.gate import Gate


class TwoQubitGate(Gate):
    """Base class for gates that operate on two qubits."""

    def __init__(self, name: str, unitary: np.ndarray, duration: float):
        if unitary.shape != (4, 4):
            raise ValueError("Unitary for a two-qubit gate must be 4x4.")
        super().__init__(name, num_qubits=2, duration=duration)
        self._unitary = unitary

    def to_unitary(self) -> np.ndarray:
        """Returns the fixed 4x4 unitary matrix for this gate."""
        return self._unitary


class CNOT(TwoQubitGate):
    """
    Controlled-NOT (CX) gate.
    Flips the target qubit (second index) if the control qubit (first index) is in state |1>.
    """

    def __init__(self):
        # CNOT matrix in the standard basis {|00>, |01>, |10>, |11>}
        unitary = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)

        # Hardware-realistic duration: CNOT is significantly slower than single-qubit gates
        super().__init__("CNOT", unitary, duration=300e-9)

    def to_qiskit_instruction(self) -> Any:
        """Returns the Qiskit CXGate equivalent."""
        return CXGate()
