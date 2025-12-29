# quantum_sim/backends/numpy_backend.py

import numpy as np
from typing import Dict, List, Optional, Any

from quantum_sim.backends.backend import QuantumBackend
# Note: Ensure GateOperation is defined/exported in quantum_sim.core.circuit
from quantum_sim.core.circuit import QuantumCircuit, GateOperation
from quantum_sim.core.noise import NoiseChannel, ThermalRelaxationChannel


class NumpyBackend(QuantumBackend):
    """
    A quantum simulation backend that uses NumPy for density matrix manipulation
    and accurately models time-dependent noise (T1/T2) and per-gate noise.
    """

    def __init__(self,
                 num_qubits: int,
                 t1_times: Optional[Dict[int, float]] = None,
                 t2_times: Optional[Dict[int, float]] = None,
                 p_ex: float = 0.0,
                 per_qubit_noise_channels: Optional[Dict[int, List[NoiseChannel]]] = None):
        """
        Initializes the backend with hardware-specific noise parameters.
        """
        self.num_total_qubits = num_qubits
        self.t1_times = t1_times if t1_times is not None else {}
        self.t2_times = t2_times if t2_times is not None else {}
        self.p_ex = p_ex
        self.per_qubit_noise_channels = per_qubit_noise_channels if per_qubit_noise_channels is not None else {}

        # Pre-instantiate ThermalRelaxationChannels for idle noise
        self._idle_noise_channels: Dict[int, ThermalRelaxationChannel] = {}
        for q_id in range(self.num_total_qubits):
            if q_id in self.t1_times and q_id in self.t2_times:
                self._idle_noise_channels[q_id] = ThermalRelaxationChannel(
                    self.t1_times[q_id], self.t2_times[q_id], self.p_ex
                )
            elif (q_id in self.t1_times and q_id not in self.t2_times) or \
                 (q_id not in self.t1_times and q_id in self.t2_times):
                raise ValueError(
                    f"Qubit {q_id} must have both T1 and T2 times specified or neither."
                )

    def _create_initial_density_matrix(self, num_qubits: int) -> np.ndarray:
        """Initializes the density matrix to |0...0><0...0|."""
        state_vec = np.zeros(2**num_qubits, dtype=complex)
        state_vec[0] = 1.0
        rho_flat = np.outer(state_vec, np.conj(state_vec))
        return rho_flat.reshape([2] * (2 * num_qubits))

    def run_circuit(self, circuit: QuantumCircuit) -> np.ndarray:
        """Executes the circuit with time-tracking and hardware-aware noise."""
        current_time = 0.0
        qubit_last_op_time: Dict[int, float] = {q_id: 0.0 for q_id in range(circuit.num_qubits)}
        current_rho_tensor = self._create_initial_density_matrix(circuit.num_qubits)
        qubit_map = {q_id: q_id for q_id in range(circuit.num_qubits)}
        # Ensure circuit._components is handled by type checking 
        for component in getattr(circuit, "_components", []):
            gate_duration = 0.0
            if isinstance(component, GateOperation):
                gate_duration = component.gate.duration
            elif isinstance(component, QuantumCircuit):
                # Safely access internal components of sub-circuits
                sub_comps = getattr(component, "_components", [])
                durations = [op.gate.duration for op in sub_comps if isinstance(op, GateOperation)]
                gate_duration = max(durations) if durations else 0.0

            # --- Apply IDLE Noise BEFORE operation ---
            time_before_op = current_time
            for q_id in range(circuit.num_qubits):
                dt_idle = time_before_op - qubit_last_op_time[q_id]
                if dt_idle > 0 and q_id in self._idle_noise_channels:
                    current_rho_tensor = self._idle_noise_channels[q_id].apply_to_density_matrix(
                        current_rho_tensor, q_id, circuit.num_qubits, dt=dt_idle
                    )
                    qubit_last_op_time[q_id] = time_before_op

            # --- Apply Unitary Operation ---
            current_rho_tensor = component.apply_to_density_matrix(current_rho_tensor, circuit.num_qubits, qubit_map)

            # --- Apply PER-GATE Noise ---
            affected_qubits = [qubit_map[local_id] for local_id in component.get_involved_qubit_local_ids()]
            for q_global_id in affected_qubits:
                if q_global_id in self.per_qubit_noise_channels:
                    for noise_channel in self.per_qubit_noise_channels[q_global_id]:
                        current_rho_tensor = noise_channel.apply_to_density_matrix(
                            current_rho_tensor, q_global_id, circuit.num_qubits, dt=gate_duration
                        )

            current_time += gate_duration
            for q_id in affected_qubits:
                qubit_last_op_time[q_id] = current_time

        # --- Final Idle Noise ---
        for q_id in range(circuit.num_qubits):
            dt_final = current_time - qubit_last_op_time[q_id]
            if dt_final > 0 and q_id in self._idle_noise_channels:
                current_rho_tensor = self._idle_noise_channels[q_id].apply_to_density_matrix(
                    current_rho_tensor, q_id, circuit.num_qubits, dt=dt_final
                )

        final_dim = 2**circuit.num_qubits
        return current_rho_tensor.reshape((final_dim, final_dim))

    def get_probabilities(self, rho_matrix: np.ndarray) -> np.ndarray:
        return np.diag(rho_matrix).real

    def get_measurements(self, rho_matrix: np.ndarray, num_shots: int) -> Dict[str, int]:
        probabilities = self.get_probabilities(rho_matrix)
        num_qubits = int(np.log2(rho_matrix.shape[0]))
        outcomes = np.random.choice(len(rho_matrix), size=num_shots, p=probabilities)
        counts: Dict[str, int] = {}
        for outcome in outcomes:
            bitstring = bin(int(outcome))[2:].zfill(num_qubits)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts
