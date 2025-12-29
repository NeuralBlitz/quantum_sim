# quantum_sim/__init__.py

"""
Quantum Simulation Engine
A high-performance density matrix simulator for NISQ-era algorithms.
"""

from .core.circuit import QuantumCircuit
from .core.parameter import Parameter
from .backend.numpy_backend import NumpyBackend

__all__ = [
    "QuantumCircuit",
    "Parameter",
    "NumpyBackend",
]
