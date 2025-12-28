# quantum_sim/core/qubit.py

class Qubit:
    """
    Represents a single quantum bit (qubit) within a quantum register or circuit.
    Each qubit has a unique ID, which typically corresponds to its index in the overall
    quantum state vector or density matrix.
    """
    def __init__(self, id: int):
        if not isinstance(id, int) or id < 0:
            raise ValueError("Qubit ID must be a non-negative integer.")
        self.id = id

    def __repr__(self) -> str:
        return f"Qubit({self.id})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Qubit):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)
