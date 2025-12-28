# quantum_sim/backends/backend.py

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from quantum_sim.core.circuit import QuantumCircuit

class QuantumBackend(ABC):
    """
    Abstract Base Class for all quantum simulation backends.
    Defines the interface for running circuits and obtaining results.
    All backends now operate on density matrices.
    """
    @abstractmethod
    def run_circuit(self, circuit: "QuantumCircuit") -> np.ndarray:
        """
        Executes the given quantum circuit and returns the final density matrix (2^N x 2^N).
        """
        pass

    @abstractmethod
    def get_probabilities(self, rho_matrix: np.ndarray) -> np.ndarray:
        """
        Calculates measurement probabilities from the diagonal of the density matrix.
        """
        pass

    @abstractmethod
    def get_measurements(self, rho_matrix: np.ndarray, num_shots: int) -> Dict[str, int]:
        """
        Simulates measurements of the quantum state for a given number of shots.
        Returns a dictionary mapping bitstring outcomes to counts.
        """
        pass
