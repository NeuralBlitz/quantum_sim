# quantum_sim/utils/expectation_value.py

import numpy as np
import networkx as nx
from typing import List
import string


class ExpectationValueCalculator:
    """
    Calculates the expectation value of observables on a density matrix.
    Includes an optimized QAOA Max-Cut energy calculator.
    """

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
        """
        Constructs the full N-qubit operator matrix using einsum.
        """
        if len(pauli_string) != len(target_ids):
            raise ValueError("Pauli string length must match target qubits.")

        full_op_list = [self.identity] * self.num_qubits
        for idx, char in enumerate(pauli_string):
            full_op_list[target_ids[idx]] = self._pauli_map[char]

        # Use ASCII for einsum indices
        chars = string.ascii_lowercase
        in_labels = [chars[i] for i in range(self.num_qubits)]
        out_labels = [chars[i + self.num_qubits] for i in range(self.num_qubits)]

        einsum_str = ",".join(f"{o}{i}" for i, o in zip(in_labels, out_labels))
        einsum_str += f"->{''.join(out_labels)}{''.join(in_labels)}"

        full_tensor = np.einsum(einsum_str, *full_op_list)
        dim = 2**self.num_qubits
        return full_tensor.reshape((dim, dim))

    def calculate_expectation_value(self, rho_matrix: np.ndarray,
                                    pauli_str: str,
                                    target_ids: List[int]) -> float:
        """
        Calculates <O> = Tr(rho * O).
        """
        dim = 2**self.num_qubits
        if rho_matrix.shape != (dim, dim):
            raise ValueError(f"Rho shape {rho_matrix.shape} mismatch for {self.num_qubits} qubits.")

        op_matrix = self._get_pauli_operator_matrix(pauli_str, target_ids)
        # trace(A @ B) is more efficient as np.sum(A * B.T)
        return np.real(np.trace(rho_matrix @ op_matrix))

    def calculate_qaoa_maxcut_energy(self, rho_matrix: np.ndarray, graph: nx.Graph) -> float:
        """
        Calculates H_C = sum_{u,v} (I - Z_u Z_v)/2.
        Optimized to handle the density matrix as a tensor to avoid huge matrix products.
        """
        total_energy = 0.0
        num_dim = 2**self.num_qubits
        
        # Reshape to tensor: rho[out_q0, out_q1, ..., in_q0, in_q1, ...]
        rho_tensor = rho_matrix.reshape([2] * (2 * self.num_qubits))

        for u, v in graph.edges():
            # We only need the expectation of Z_u * Z_v
            # Trace out all qubits except u and v
            keep = [u, v]
            trace_indices = [i for i in range(self.num_qubits) if i not in keep]
            
            # Partial trace for qubits u and v
            # Sum over the input and output indices of all other qubits
            einsum_indices = list(range(2 * self.num_qubits))
            for i in trace_indices:
                einsum_indices[i + self.num_qubits] = einsum_indices[i]
            
            # Contract to get a 4x4 reduced density matrix for qubits u and v
            reduced_rho = np.einsum(rho_tensor, einsum_indices, [u, v, u + self.num_qubits, v + self.num_qubits])
            reduced_rho = reduced_rho.reshape((4, 4))
            
            # Z_u Z_v matrix in 4x4: diag(1, -1, -1, 1)
            zz_matrix = np.diag([1, -1, -1, 1]).astype(complex)
            zz_exp = np.real(np.trace(reduced_rho @ zz_matrix))
            
            total_energy += 0.5 * (1.0 - zz_exp)

        return total_energy
