# quantum_sim/core/qubit.py


class Qubit:
    """
    Represents a single quantum bit (qubit) within a quantum register or circuit.
    Each qubit has a unique ID, which typically corresponds to its index in the overall
    quantum state vector or density matrix.
    """

    def __init__(self, qubit_id: int):
        if not isinstance(qubit_id, int) or qubit_id < 0:
            raise ValueError("Qubit ID must be a non-negative integer.")
        self.id = qubit_id

    def __repr__(self) -> str:
        return f"Qubit({self.id})"

    def __eq__(self, other: object) -> bool:
        """Two qubits are considered equal if they share the same ID."""
        if not isinstance(other, Qubit):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on ID to allow unique qubit tracking in sets and registers."""
        return hash(self.id)
