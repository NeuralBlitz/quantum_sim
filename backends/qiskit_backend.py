# quantum_sim/backends/qiskit_backend.py

import numpy as np
from typing import Dict, List
from qiskit import QuantumCircuit as QiskitQuantumCircuit, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

from quantum_sim.backends.backend import QuantumBackend
from quantum_sim.core.circuit import QuantumCircuit
from quantum_sim.core.parameter import Parameter
from quantum_sim.core.noise import NoiseChannel, DepolarizingChannel
from quantum_sim.gates.single_qubit_gates import Hadamard, PauliX
from quantum_sim.gates.two_qubit_gates import CNOT
from quantum_sim.gates.parametric_gates import RX, RZ


class QiskitBackend(QuantumBackend):
    """
    A quantum simulation backend that leverages Qiskit's AerSimulator for execution.
    Supports custom noise models, including time-dependent T1/T2 relaxation.
    """

    def __init__(self,
                 num_qubits: int,
                 t1_times: Dict[int, float] = None,
                 t2_times: Dict[int, float] = None,
                 p_ex: float = 0.0,
                 per_qubit_noise_channels: Dict[int, List[NoiseChannel]] = None,
                 gate_durations: Dict[str, float] = None
                 ):
        """
        Initializes the Qiskit-based backend with a specified noise model.
        """
        self.simulator = AerSimulator()
        self.noise_model = NoiseModel()
        self.num_total_qubits = num_qubits
        self.p_ex = p_ex
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

    def _build_qiskit_noise_model(self, t1_times, t2_times, p_ex, custom_noise_channels):
        """Constructs a Qiskit NoiseModel from our custom noise definitions."""
        if custom_noise_channels:
            for q_global_id, channels in custom_noise_channels.items():
                for channel in channels:
                    if isinstance(channel, DepolarizingChannel):
                        p_err = channel.p
                        # Apply to single-qubit basis gates
                        target_gates = ['u1', 'u2', 'u3', 'h', 'x', 'rx', 'ry', 'rz']
                        self.noise_model.add_quantum_error(depolarizing_error(p_err, 1), target_gates, [q_global_id])

                        # Apply to two-qubit CNOTs involving this qubit
                        for other_q_id in range(self.num_total_qubits):
                            if q_global_id != other_q_id:
                                self.noise_model.add_quantum_error(
                                    depolarizing_error(p_err, 2), ['cx'], [q_global_id, other_q_id]
                                )

                        self.noise_model.add_basis_gates(target_gates + ['cx'])

        if t1_times and t2_times:
            our_gate_names = [name for name, dur in self.gate_durations.items() if dur > 0]
            qiskit_gate_map = {"Hadamard": "h", "PauliX": "x", "CNOT": "cx", "RX": "rx", "RZ": "rz"}

            for q_id in range(self.num_total_qubits):
                if q_id in t1_times and q_id in t2_times:
                    t1, t2 = t1_times[q_id], t2_times[q_id]
                    for our_name in our_gate_names:
                        qiskit_name = qiskit_gate_map.get(our_name)
                        if qiskit_name:
                            duration_ns = self.gate_durations[our_name] * 1e9
                            error = thermal_relaxation_error(t1, t2, p_ex, duration=duration_ns, unit='ns')
                            self.noise_model.add_quantum_error(error, [qiskit_name], [q_id])
                            self.noise_model.add_basis_gates([qiskit_name])

        if self.noise_model.errors():
            self.simulator = AerSimulator(noise_model=self.noise_model)

    def run_circuit(self, circuit: QuantumCircuit) -> np.ndarray:
        """Translates and runs our circuit on the AerSimulator."""
        qreg = QuantumRegister(circuit.num_qubits, 'q')
        qiskit_qc = QiskitQuantumCircuit(qreg)
        qubit_map = {q_id: q_id for q_id in range(circuit.num_qubits)}

        circuit.get_qiskit_circuit_instructions(qiskit_qc, qubit_map)
        qiskit_qc.save_density_matrix()

        job = self.simulator.run(qiskit_qc, shots=1, basis_gates=self.noise_model.basis_gates)
        result = job.result()
        return np.array(result.get_density_matrix(qiskit_qc))

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
