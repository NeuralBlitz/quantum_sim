# quantum_sim/utils/expectation_value.py

import numpy as np
import networkx as nx
from typing import List, Tuple
import string

class ExpectationValueCalculator:
    """
    Calculates the expectation value of an observable (represented by a Pauli string)
    on a given quantum density matrix. Now includes QAOA Max-Cut energy calculation.
    """
    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.pauli_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.pauli_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.pauli_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.identity = np.array([[1, 0], [0, 1]], dtype=complex)

        self._pauli_map = {'I': self.identity, 'X': self.pauli_x, 'Y': self.pauli_y, 'Z': self.pauli_z}

    def _get_pauli_operator_matrix(self, pauli_string: str, target_qubits_global_ids: List[int]) -> np.ndarray:
        """
        Constructs the full N-qubit operator matrix (2^N x 2^N) for a given Pauli string
        acting on specific qubits, using np.einsum for efficiency.
        """
        if len(pauli_string) != len(target_qubits_global_ids):
            raise ValueError("Pauli string length must match the number of target qubits.")

        full_op_tensor_list = [self.identity] * self.num_qubits

        for i, pauli_char in enumerate(pauli_string):
            if pauli_char not in self._pauli_map:
                raise ValueError(f"Invalid Pauli character: {pauli_char}. Must be I, X, Y, or Z.")
            full_op_tensor_list[target_qubits_global_ids[i]] = self._pauli_map[pauli_char]

        input_labels = [string.ascii_lowercase[i] for i in range(self.num_qubits)]
        output_labels = [string.ascii_uppercase[i] for i in range(self.num_qubits)]

        op_einsum_str_parts = []
        for i in range(self.num_qubits):
            op_einsum_str_parts.append(f"{output_labels[i]}{input_labels[i]}")
        
        einsum_equation = ",".join(op_einsum_str_parts) + "->" + "".join(output_labels) + "".join(input_labels)

        full_op_tensor = np.einsum(einsum_equation, *full_op_tensor_list)
        full_op_matrix = full_op_tensor.reshape((2**self.num_qubits, 2**self.num_qubits))
        
        return full_op_matrix

    def calculate_expectation_value(self, rho_matrix: np.ndarray,
                                    observable_pauli_string: str,
                                    observable_target_qubits_global_ids: List[int]) -> float:
        """
        Calculates the expectation value <O> = Tr(rho * O) of an observable O
        on the given density matrix rho.
        """
        if rho_matrix.shape != (2**self.num_qubits, 2**self.num_qubits):
            raise ValueError(f"Density matrix shape ({rho_matrix.shape}) must match "
                             f"({2**self.num_qubits}, {2**self.num_qubits}) for num_qubits={self.num_qubits}.")

        observable_matrix = self._get_pauli_operator_matrix(observable_pauli_string, observable_target_qubits_global_ids)
        
        expectation_value = np.trace(rho_matrix @ observable_matrix)
        
        return np.real(expectation_value)

    def calculate_qaoa_maxcut_energy(self, rho_matrix: np.ndarray, graph: nx.Graph) -> float:
        """
        Calculates the expectation value of the Max-Cut Hamiltonian
        H_C = sum_edges (I - Z_u Z_v)/2.
        QAOA minimizes this expectation value, which corresponds to maximizing the cut.
        """
        total_expectation = 0.0
        for u, v in graph.edges():
            pauli_string_list = ['I'] * self.num_qubits
            pauli_string_list[u] = 'Z'
            pauli_string_list[v] = 'Z'
            observable_string = "".join(pauli_string_list)
            
            zz_expectation = self.calculate_expectation_value(rho_matrix, observable_string, list(range(self.num_qubits)))
            
            total_expectation += 0.5 * (1.0 - zz_expectation)
            
        return total_expectation
