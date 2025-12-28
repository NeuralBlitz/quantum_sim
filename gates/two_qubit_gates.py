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

class CNOT(TwoQubitGate):
    """Controlled-NOT gate flips the target qubit if the control is 1."""
    def __init__(self):
        unitary = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]])
        super().__init__("CNOT", unitary, duration=300e-9) # 300 ns duration

    def to_qiskit_instruction(self) -> Any:
        return CXGate()
