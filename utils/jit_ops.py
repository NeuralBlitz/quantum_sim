# quantum_sim/utils/jit_ops.py

import numpy as np
from numba import njit, complex128, int32, prange


@njit(complex128[:, :, :, :](complex128[:, :, :, :], complex128[:, :], complex128[:, :], str), cache=True)
def _jit_einsum_apply(rho_tensor, unitary, unitary_dag, equation):
    """Internal JIT function. Numba supports np.einsum if the string is passed in."""
    return np.einsum(equation, unitary, rho_tensor, unitary_dag)


def apply_unitary_to_rho(rho_tensor: np.ndarray, unitary: np.ndarray,
                         target_qubits: np.ndarray, num_total_qubits: int) -> np.ndarray:
    """
    Python wrapper to handle string generation (which Numba can't do) 
    before calling the JIT-accelerated contraction.
    """
    unitary_dag = np.conj(unitary.T)
    num_gate_qubits = len(target_qubits)
    
    # Generate indices: row indices are even, col indices are odd
    row_idx = [f"a{k}" for k in range(num_total_qubits)]
    col_idx = [f"b{k}" for k in range(num_total_qubits)]
    
    input_rho_labels = []
    for r, c in zip(row_idx, col_idx):
        input_rho_labels.extend([r, c])
    input_rho_str = "".join(input_rho_labels)
    
    # Generate new labels for the affected qubits
    new_rows = [f"x{k}" for k in range(num_gate_qubits)]
    new_cols = [f"y{k}" for k in range(num_gate_qubits)]
    
    u_in = "".join([row_idx[q] for q in target_qubits])
    u_out = "".join(new_rows)
    
    udag_in = "".join([col_idx[q] for q in target_qubits])
    udag_out = "".join(new_cols)
    
    output_labels = list(input_rho_labels)
    for i, q_idx in enumerate(target_qubits):
        output_labels[2 * q_idx] = new_rows[i]
        output_labels[2 * q_idx + 1] = new_cols[i]
    output_rho_str = "".join(output_labels)
    
    equation = f"{u_out}{u_in},{input_rho_str},{udag_in}{udag_out}->{output_rho_str}"
    
    return _jit_einsum_apply(rho_tensor, unitary, unitary_dag, equation)


@njit(complex128[:, :, :, :](complex128[:, :, :, :], complex128[:, :, :], int32, int32), 
      cache=True, parallel=True)
def jit_apply_kraus_to_rho(rho_tensor: np.ndarray, kraus_ops: np.ndarray,
                           target_idx: int, num_total_qubits: int) -> np.ndarray:
    """
    Numba-accelerated Kraus map application. 
    Parallelized over the Kraus operators (e.g., for depolarizing or amplitude damping).
    """
    # Initialize result
    res_rho = np.zeros_like(rho_tensor)
    
    # Since we can't easily build dynamic strings in @njit, we use a 
    # specific implementation for 1-qubit noise channels (most common in NISQ)
    # Using manual loops or fixed einsum patterns for speed
    num_kraus = kraus_ops.shape[0]
    
    for k in prange(num_kraus):
        e_k = kraus_ops[k]
        e_k_dag = np.conj(e_k.T)
        
        # We manually perform the 1-qubit contraction logic here to bypass 
        # Numba's string limitations while keeping it JIT-fast
        # (Alternatively, pass the equation string from a wrapper)
        res_rho += _apply_single_kraus_term(rho_tensor, e_k, e_k_dag, target_idx)
            
    return res_rho

@njit(cache=True)
def _apply_single_kraus_term(rho, e, edag, q):
    """Helper for internal 1-qubit Kraus contraction."""
    # Simplified logic: 1-qubit Kraus is always the same 'shape' of contraction
    # This effectively implements the sum_k E_k rho E_k^dag
    return np.einsum("xy, ...y...Y..., YZ -> ...x...Z...", e, rho, edag)

