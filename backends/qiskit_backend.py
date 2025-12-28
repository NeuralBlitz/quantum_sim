# quantum_sim/backends/qiskit_backend.py

import numpy as np
from typing import Dict, Any, List
from qiskit import QuantumCircuit as QiskitQuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.result import Result
from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

from quantum_sim.backends.backend import QuantumBackend
from quantum_sim.core.circuit import QuantumCircuit
from quantum_sim.core.parameter import Parameter # For accessing gate duration from dummy gate instance
from quantum_sim.core.noise import NoiseChannel, DepolarizingChannel, ThermalRelaxationChannel # NEW
from quantum_sim.gates.single_qubit_gates import Hadamard, PauliX
from quantum_sim.gates.two_qubit_gates import CNOT
from quantum_sim.gates.parametric_gates import RX, RZ

class QiskitBackend(QuantumBackend):
    """
    A quantum simulation backend that leverages Qiskit's AerSimulator for execution.
    Now supports custom noise models, including time-dependent T1/T2 relaxation and depolarizing errors.
    """
    def __init__(self, 
                 num_qubits: int, 
                 t1_times: Dict[int, float] = None, 
                 t2_times: Dict[int, float] = None, 
                 p_ex: float = 0.0, 
                 per_qubit_noise_channels: Dict[int, List[NoiseChannel]] = None,
                 gate_durations: Dict[str, float] = None # To map our gate names to durations
                ):
        self.simulator = AerSimulator()
        self.noise_model = NoiseModel()
        self.num_total_qubits = num_qubits
        self.gate_durations = gate_durations if gate_durations is not None else self._get_default_gate_durations()

        self._build_qiskit_noise_model(t1_times, t2_times, p_ex, per_qubit_noise_channels)

    def _get_default_gate_durations(self) -> Dict[str, float]:
        """Collects default durations from our gate classes."""
        return {
            "Hadamard": Hadamard().duration,
            "PauliX": PauliX().duration,
            "CNOT": CNOT().duration,
            "RX": RX(Parameter("dummy")).duration,
            "RZ": RZ(Parameter("dummy")).duration,
        }

    def _build_qiskit_noise_model(self, t1_times: Dict[int, float], t2_times: Dict[int, float], p_ex: float, custom_noise_channels: Dict[int, List[NoiseChannel]]):
        
        # --- Apply per-qubit, non-time-dependent noise (e.g., Depolarizing) ---
        if custom_noise_channels:
            for q_global_id, channels in custom_noise_channels.items():
                for channel in channels:
                    if isinstance(channel, DepolarizingChannel):
                        p_error = channel.p
                        # Map to Qiskit's standard basis gates for single and two-qubit errors
                        # Qiskit maps 'h', 'x', 'rx', 'ry', 'rz' directly to basis gates u1/u2/u3
                        self.noise_model.add_quantum_error(depolarizing_error(p_error, 1), ['u1', 'u2', 'u3', 'h', 'x', 'rx', 'ry', 'rz'], [q_global_id])
                        
                        # Apply to CNOTs. Qiskit's depolarizing_error can be applied to 2-qubit gates
                        # It will create errors for all CNOTs that involve this qubit.
                        self# For full rigor, would add for specific pairs. For global, easier.
                        # This covers all 2Q operations involving q_global_id as first or second qubit.
                        for other_q_id in range(self.num_total_qubits):
                            if q_global_id != other_q_id:
                                self.noise_model.add_quantum_error(depolarizing_error(p_error, 2), ['cx'], [q_global_id, other_q_id])
                        
                        self.noise_model.add_basis_gates(['u1', 'u2', 'u3', 'h', 'x', 'rx', 'ry', 'rz', 'cx'])


        # --- Apply Thermal Relaxation noise ---
        if t1_times and t2_times:
            # Qiskit's `thermal_relaxation_error` automatically handles idle noise if basis_gates are set
            # and `duration` is provided to `add_quantum_error` for specific gates.
            
            # Collect all our custom gate names that have durations
            our_gate_names_with_durations = [name for name, dur in self.gate_durations.items() if dur > 0]
            
            # Map our gate names to Qiskit's corresponding basis gate names
            qiskit_gate_name_map = {
                "Hadamard": "h", "PauliX": "x", "CNOT": "cx", "RX": "rx", "RZ": "rz" # Add others as needed
            }

            for q_id in range(self.num_total_qubits):
                if q_id in t1_times and q_id in t2_times:
                    t1 = t1_times[q_id]
                    t2 = t2_times[q_id]
                    
                    for our_gate_name in our_gate_names_with_durations:
                        qiskit_gate_name = qiskit_gate_name_map.get(our_gate_name, None)
                        if qiskit_gate_name:
                            duration_s = self.gate_durations[our_gate_name]
                            duration_ns = duration_s * 1e9 # Convert to ns for Qiskit's duration mapping

                            error = thermal_relaxation_error(t1, t2, self.p_ex, duration=duration_ns, unit='ns')
                            
                            # Add error for this qubit on the specific Qiskit gate type
                            self.noise_model.add_quantum_error(error, [qiskit_gate_name], [q_id])
                            
                            # Ensure these basis gates are known to the noise model
                            self.noise_model.add_basis_gates([qiskit_gate_name])

                    # Add an 'idle' error to account for idle time for qubits (Qiskit handles this well)
                    # thermal_relaxation_error for 'id' gate or general idle
                    # This is implicitly handled if all other gates have durations.
                    # Qiskit's AerSimulator automatically models idle time.

        if self.noise_model.errors():
            self.simulator = AerSimulator(noise_model=self.noise_model)
            print(f"QiskitBackend initialized with noise model: {self.noise_model}")

    def run_circuit(self, circuit: QuantumCircuit) -> np.ndarray:
        qreg = QuantumRegister(circuit.num_qubits, 'q')
        qiskit_qc = QiskitQuantumCircuit(qreg)
        
        qubit_map = {q_id: q_id for q_id in range(circuit.num_qubits)}

        circuit.get_qiskit_circuit_instructions(qiskit_qc, qubit_map)

        qiskit_qc.save_density_matrix()
        
        job = self.simulator.run(qiskit_qc, shots=1, basis_gates=self.noise_model.basis_gates)
        result = job.result()
        density_matrix = result.get_density_matrix(qiskit_qc)
        return np.array(density_matrix)

    def get_probabilities(self, rho_matrix: np.ndarray) -> np.ndarray:
        return np.diag(rho_matrix).real

    def get_measurements(self, rho_matrix: np.ndarray, num_shots: int) -> Dict[str, int]:
        probabilities = self.get_probabilities(rho_matrix)
        num_qubits = int(np.log2(rho_matrix.shape[0]))
        
        outcomes = np.random.choice(len(rho_matrix), size=num_shots, p=probabilities)
        
        counts = {}
        for outcome in outcomes:
            bitstring = bin(outcome)[2:].zfill(num_qubits)
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts
