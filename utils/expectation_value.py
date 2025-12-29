# quantum_sim/utils/expectation_value.py

import numpy as np
import networkx as nx
from typing import List
import string


class ExpectationValueCalculator:
    """Calculates the expectation value of observables on a density matrix."""

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.array([[1, 0], [0, 1]], dtype=complex)
        self._pauli_map = {
            'I': self.identity,
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': self.pauli_z
        }

    def _get_pauli_operator_matrix(self, pauli_string: str, target_ids: List[int]) -> np.ndarray:
        if len(pauli_string) != len(target_ids):
            raise ValueError("String length mismatch.")
        full_op_list = [self.identity] * self.num_qubits
        for idx, char in enumerate(pauli_string):
            full_op_list[target_ids[idx]] = self._pauli_map[char]
        chars = string.ascii_lowercase
        in_labels = [chars[i] for i in range(self.num_qubits)]
        out_labels = [chars[i + self.num_qubits] for i in range(self.num_qubits)]
        eqn = ",".join(f"{o}{i}" for i, o in zip(in_labels, out_labels))
        eqn += f"->{''.join(out_labels)}{''.join(in_labels)}"
        full_tensor = np.einsum(eqn, *full_op_list)
        dim = 2**self.num_qubits
        return full_tensor.reshape((dim, dim))

    def calculate_expectation_value(self, rho_matrix: np.ndarray, pauli_str: str,
                                    target_ids: List[int]) -> float:
        op_matrix = self._get_pauli_operator_matrix(pauli_str, target_ids)
        return np.real(np.trace(rho_matrix @ op_matrix))

    def calculate_qaoa_maxcut_energy(self, rho_matrix: np.ndarray, graph: nx.Graph) -> float:
        total_energy = 0.0
        rho_tensor = rho_matrix.reshape([2] * (2 * self.num_qubits))
        for u, v in graph.edges():
            trace_indices = [i for i in range(self.num_qubits) if i not in [u, v]]
            einsum_idx = list(range(2 * self.num_qubits))
            for i in trace_indices:
                einsum_idx[i + self.num_qubits] = einsum_idx[i]
            red_rho = np.einsum(rho_tensor, einsum_idx, [u, v, u + self.num_qubits, v + self.num_qubits])
            red_rho = red_rho.reshape((4, 4))
            zz_matrix = np.diag([1, -1, -1, 1]).astype(complex)
            zz_exp = np.real(np.trace(red_rho @ zz_matrix))
            total_energy += 0.5 * (1.0 - zz_exp)
        return total_energy
