# quantum_sim/core/noise.py

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple
from quantum_sim.utils.jit_ops import jit_apply_kraus_to_rho
import string

class NoiseChannel(ABC):
    """
    Abstract Base Class for Kraus-based noise channels.
    Defines the interface for channels that apply noise to a density matrix.
    """
    @abstractmethod
    def get_kraus_operators(self, dt: float = 0.0) -> List[np.ndarray]:
        """
        Returns the list of Kraus matrices {E_k} for a single qubit,
        potentially dependent on a time duration dt.
        """
        pass

    @abstractmethod
    def apply_to_density_matrix(self, rho_tensor: np.ndarray, target_qubit_idx: int, num_total_qubits: int, dt: float = 0.0) -> np.ndarray:
        """
        Applies the noise channel to a density matrix tensor (2N-dimensional)
        on a specific target qubit.
        dt: The time duration over which to apply time-dependent noise.
        """
        pass

class DepolarizingChannel(NoiseChannel):
    def __init__(self, p: float):
        if not (0 <= p <= 1):
            raise ValueError("Error probability p for DepolarizingChannel must be in [0, 1].")
        self.p = p

    def get_kraus_operators(self, dt: float = 0.0) -> List[np.ndarray]:
        p = self.p
        I = np.array([[1, 0], [0, 1]], dtype=complex)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        
        kraus_ops = [
            np.sqrt(1 - 0.75 * p) * I,
            np.sqrt(p / 4) * X,
            np.sqrt(p / 4) * Y,
            np.sqrt(p / 4) * Z
        ]
        return [op for op in kraus_ops if np.linalg.norm(op) > 1e-12]

    def apply_to_density_matrix(self, rho_tensor: np.ndarray, target_qubit_idx: int, num_total_qubits: int, dt: float = 0.0) -> np.ndarray:
        kraus_ops = self.get_kraus_operators(dt=0.0)
        
        kraus_ops_array = np.array(kraus_ops, dtype=complex)
        
        return jit_apply_kraus_to_rho(rho_tensor, kraus_ops_array, target_qubit_idx, num_total_qubits)


class ThermalRelaxationChannel(NoiseChannel):
    """
    Implements the T1/T2 thermal relaxation and dephasing channel.
    Models energy decay (T1) towards a thermal state and loss of coherence (T2).
    Uses a canonical 3-operator Kraus set for P_ex=0, assuming T2 <= T1 for accuracy.
    """
    def __init__(self, t1: float, t2: float, p_ex: float = 0.0):
        if not isinstance(t1, (int, float)) or t1 <= 0:
            raise ValueError("T1 relaxation time must be a positive number.")
        if not isinstance(t2, (int, float)) or t2 <= 0:
            raise ValueError("T2 dephasing time must be a positive number.")
        if not (0 <= p_ex <= 1):
            raise ValueError("Excited state population P_ex must be in [0, 1].")
        if t2 > 2 * t1:
            raise ValueError(f"Physical constraint violation: T2 ({t2*1e6:.1f}us) must be <= 2*T1 ({t1*1e6:.1f}us).")
        if t2 < t1:
            print(f"Warning: T2 ({t2*1e6:.1f}us) is less than T1 ({t1*1e6:.1f}us). "
                  "The simplified 3-Kraus set for P_ex=0 might be less accurate for this T1/T2 ratio. "
                  "This model correctly captures decay to |0>.")

        self.t1 = t1
        self.t2 = t2
        self.p_ex = p_ex

    def get_kraus_operators(self, dt: float) -> List[np.ndarray]:
        if dt < 0:
            raise ValueError("Time duration dt must be non-negative.")
        if dt == 0:
            return [np.eye(2, dtype=complex)]

        gamma_1 = np.exp(-dt / self.t1)
        gamma_2 = np.exp(-dt / self.t2)
        
        term_e2_sqrt = gamma_1 - gamma_2
        if term_e2_sqrt < 0:
             term_e2_sqrt = 0.0
        
        E0 = np.array([[1, 0], [0, np.sqrt(gamma_2)]], dtype=complex)
        E1 = np.array([[0, np.sqrt(1 - gamma_1)], [0, 0]], dtype=complex)
        E2 = np.array([[0, 0], [0, np.sqrt(term_e2_sqrt)]], dtype=complex)

        return [op for op in [E0, E1, E2] if np.linalg.norm(op) > 1e-12]

    def apply_to_density_matrix(self, rho_tensor: np.ndarray, target_qubit_idx: int, num_total_qubits: int, dt: float = 0.0) -> np.ndarray:
        kraus_ops = self.get_kraus_operators(dt=dt)
        
        kraus_ops_array = np.array(kraus_ops, dtype=complex)
        
        return jit_apply_kraus_to_rho(rho_tensor, kraus_ops_array, target_qubit_idx, num_total_qubits)
