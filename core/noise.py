# quantum_sim/core/noise.py

from abc import ABC, abstractmethod
from typing import List
import numpy as np

from quantum_sim.utils.jit_ops import jit_apply_kraus_to_rho


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
    def apply_to_density_matrix(self, rho_tensor: np.ndarray, target_qubit_idx: int,
                                num_total_qubits: int, dt: float = 0.0) -> np.ndarray:
        """
        Applies the noise channel to a density matrix tensor (2N-dimensional)
        on a specific target qubit.
        """
        pass


class DepolarizingChannel(NoiseChannel):
    """
    Models a depolarizing error channel where the state is replaced
    by the maximally mixed state with probability p.
    """

    def __init__(self, p: float):
        if not (0 <= p <= 1):
            raise ValueError("Error probability p for DepolarizingChannel must be in [0, 1].")
        self.p = p

    def get_kraus_operators(self, dt: float = 0.0) -> List[np.ndarray]:
        """Returns the {I, X, Y, Z} Kraus operators."""
        prob = self.p
        # E741 Fix: Renamed 'I' to 'eye_mat' to avoid ambiguous variable name
        eye_mat = np.array([[1, 0], [0, 1]], dtype=complex)
        sig_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sig_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sig_z = np.array([[1, 0], [0, -1]], dtype=complex)

        kraus_ops = [
            np.sqrt(1 - 0.75 * prob) * eye_mat,
            np.sqrt(prob / 4) * sig_x,
            np.sqrt(prob / 4) * sig_y,
            np.sqrt(prob / 4) * sig_z
        ]
        return [op for op in kraus_ops if np.linalg.norm(op) > 1e-12]

    def apply_to_density_matrix(self, rho_tensor: np.ndarray, target_qubit_idx: int,
                                num_total_qubits: int, dt: float = 0.0) -> np.ndarray:
        kraus_ops = self.get_kraus_operators(dt=0.0)
        kraus_ops_array = np.array(kraus_ops, dtype=complex)
        return jit_apply_kraus_to_rho(rho_tensor, kraus_ops_array, target_qubit_idx, num_total_qubits)


class ThermalRelaxationChannel(NoiseChannel):
    """
    Implements the T1/T2 thermal relaxation and dephasing channel.
    Models energy decay (T1) and loss of coherence (T2).
    """

    def __init__(self, t1: float, t2: float, p_ex: float = 0.0):
        if not isinstance(t1, (int, float)) or t1 <= 0:
            raise ValueError("T1 relaxation time must be a positive number.")
        if not isinstance(t2, (int, float)) or t2 <= 0:
            raise ValueError("T2 dephasing time must be a positive number.")
        if not (0 <= p_ex <= 1):
            raise ValueError("Excited state population P_ex must be in [0, 1].")
        if t2 > 2 * t1:
            raise ValueError(f"Constraint violation: T2 ({t2*1e6:.1f}us) must be <= 2*T1 ({t1*1e6:.1f}us).")

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

        term_e2_sqrt = max(0.0, gamma_1 - gamma_2)

        ops = [
            np.array([[1, 0], [0, np.sqrt(gamma_2)]], dtype=complex),
            np.array([[0, np.sqrt(1 - gamma_1)], [0, 0]], dtype=complex),
            np.array([[0, 0], [0, np.sqrt(term_e2_sqrt)]], dtype=complex)
        ]

        return [op for op in ops if np.linalg.norm(op) > 1e-12]

    def apply_to_density_matrix(self, rho_tensor: np.ndarray, target_qubit_idx: int,
                                num_total_qubits: int, dt: float = 0.0) -> np.ndarray:
        kraus_ops = self.get_kraus_operators(dt=dt)
        kraus_ops_array = np.array(kraus_ops, dtype=complex)
        return jit_apply_kraus_to_rho(rho_tensor, kraus_ops_array, target_qubit_idx, num_total_qubits)
