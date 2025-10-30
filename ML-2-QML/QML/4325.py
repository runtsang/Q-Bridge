import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from typing import Iterable, Sequence

# ----------------------------------------------------------------------
# Quantum circuit that mirrors the classical fully‑connected layer
# ----------------------------------------------------------------------
class HybridFCL:
    """
    A parameterised quantum circuit that can act as a drop‑in replacement
    for the classical HybridFCL.  The circuit consists of a Hadamard layer,
    a trainable Ry rotation (one per qubit), and measurement in the computational
    basis.  The expectation value of the bit‑string is returned as the network output.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 1024):
        self.n_qubits = n_qubits
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Parameterised circuit
        self.theta = Parameter("θ")
        self.circuit = QuantumCircuit(n_qubits)
        self.circuit.h(range(n_qubits))
        self.circuit.barrier()
        self.circuit.ry(self.theta, range(n_qubits))
        self.circuit.measure_all()

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the circuit for each value in *thetas* and return the
        expectation value of the measured bit‑string.
        """
        job = execute(
            self.circuit,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: t} for t in thetas],
        )
        result = job.result()
        counts = result.get_counts()
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        expectation = np.sum(states * probs)
        return np.array([expectation])


# ----------------------------------------------------------------------
# Quantum regression dataset (from QuantumRegression.py)
# ----------------------------------------------------------------------
def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample quantum states of the form cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)

    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1

    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels


__all__ = ["HybridFCL", "generate_superposition_data"]
