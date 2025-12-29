# quantum_sim/utils/jit_ops.py

import numpy as np
from numba import njit, complex128, int32, prange


@njit(complex128[:, :, :, :](complex128[:, :, :, :], complex128[:, :], complex128[:, :], str), cache=True)
def _jit_einsum_apply(rho_tensor, unitary, unitary_dag, equation):
    return np.einsum(equation, unitary, rho_tensor, unitary_dag)


def apply_unitary_to_rho(rho_tensor: np.ndarray, unitary: np.ndarray,
                         target_qubits: np.ndarray, num_total_qubits: int) -> np.ndarray:
    unitary_dag = np.conj(unitary.T)
    num_gate_qubits = len(target_qubits)
    row_idx = [f"a{k}" for k in range(num_total_qubits)]
    col_idx = [f"b{k}" for k in range(num_total_qubits)]
    in_rho_labels = []
    for r, c in zip(row_idx, col_idx):
        in_rho_labels.extend([r, c])
    in_rho_str = "".join(in_rho_labels)
    new_rows = [f"x{k}" for k in range(num_gate_qubits)]
    new_cols = [f"y{k}" for k in range(num_gate_qubits)]
    u_in = "".join([row_idx[q] for q in target_qubits])
    u_out = "".join(new_rows)
    udag_in = "".join([col_idx[q] for q in target_qubits])
    udag_out = "".join(new_cols)
    out_labels = list(in_rho_labels)
    for i, q_idx in enumerate(target_qubits):
        out_labels[2 * q_idx] = new_rows[i]
        out_labels[2 * q_idx + 1] = new_cols[i]
    out_rho_str = "".join(out_labels)
    eqn = f"{u_out}{u_in},{in_rho_str},{udag_in}{udag_out}->{out_rho_str}"
    return _jit_einsum_apply(rho_tensor, unitary, unitary_dag, eqn)


@njit(complex128[:, :, :, :](complex128[:, :, :, :], complex128[:, :, :], int32, int32),
      cache=True, parallel=True)
def jit_apply_kraus_to_rho(rho_tensor: np.ndarray, kraus_ops: np.ndarray,
                           target_idx: int, num_total_qubits: int) -> np.ndarray:
    res_rho = np.zeros_like(rho_tensor)
    for k in prange(kraus_ops.shape[0]):
        e_k = kraus_ops[k]
        e_k_dag = np.conj(e_k.T)
        res_rho += _apply_single_kraus_term(rho_tensor, e_k, e_k_dag, target_idx)
    return res_rho


@njit(cache=True)
def _apply_single_kraus_term(rho, e, edag, q):
    return np.einsum("xy, ...y...Y..., YZ -> ...x...Z...", e, rho, edag)
