# quantum_sim/core/register.py

from typing import List, Iterator
from quantum_sim.core.qubit import Qubit


class Register:
    """
    Represents a collection of qubits. This is useful for defining operations
    that act on a block of qubits, or for sub-circuits that have their own
    local qubit indexing.
    """

    def __init__(self, size: int, offset: int = 0):
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Register size must be a positive integer.")
        if not isinstance(offset, int) or offset < 0:
            raise ValueError("Register offset must be a non-negative integer.")

        self.size = size
        self.offset = offset
        # Fixed E741: Renamed 'i' to 'idx' for clarity and lint compliance
        self.qubits: List[Qubit] = [Qubit(offset + idx) for idx in range(size)]

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> Qubit:
        if not isinstance(index, int):
            raise TypeError("Index must be an integer.")
        if not (0 <= index < self.size):
            raise IndexError("Register index out of bounds.")
        return self.qubits[index]

    def __iter__(self) -> Iterator[Qubit]:
        return iter(self.qubits)

    def __repr__(self) -> str:
        qubit_ids = ", ".join(str(q.id) for q in self.qubits)
        return f"Register({self.size} qubits, IDs: [{qubit_ids}])"

    def get_qubit_ids(self) -> List[int]:
        """Returns a list of integer IDs for all qubits in the register."""
        return [q.id for q in self.qubits]
