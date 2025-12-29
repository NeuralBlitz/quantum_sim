from .circuit import QuantumCircuit
from .register import Register
from .parameter import Parameter
class QuantumCircuit:
    def __init__(self, num_qubits: int, name: str = "Circuit"):
        self.num_qubits = num_qubits
        self.name = name
        self.operations = []
        self._components = [] # Ensure this exists if backends call it
