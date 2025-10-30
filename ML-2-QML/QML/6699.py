import numpy as np
import qiskit
from qiskit import QuantumCircuit, execute, Aer
from qiskit.circuit.random import random_circuit

class SelfAttention:
    """Quantum self‑attention that combines a quanvolution‑style filter with an
    attention‑style entangling block.  The interface mimics the classical
    implementation so the two can be swapped during experimentation."""
    def __init__(self, kernel_size: int = 2, threshold: float = 127, shots: int = 1024):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.threshold = threshold
        self.shots = shots
        self.backend = Aer.get_backend("qasm_simulator")

        # Build the convolutional sub‑circuit
        self._conv_circuit = QuantumCircuit(self.n_qubits)
        self.theta = [qiskit.circuit.Parameter(f"theta{i}") for i in range(self.n_qubits)]
        for i in range(self.n_qubits):
            self._conv_circuit.rx(self.theta[i], i)
        self._conv_circuit.barrier()
        self._conv_circuit += random_circuit(self.n_qubits, 2)
        self._conv_circuit.measure_all()

    def _build_attention(self, rotation_params: np.ndarray, entangle_params: np.ndarray) -> QuantumCircuit:
        """Create a new circuit that attaches the attention block to the
        convolutional sub‑circuit."""
        circ = QuantumCircuit(self.n_qubits)
        circ += self._conv_circuit

        # Apply rotation gates (treated as the query transformation)
        for i, angle in enumerate(rotation_params):
            circ.ry(angle, i)

        # Entangling gates (treated as the key transformation)
        for i in range(self.n_qubits - 1):
            circ.crx(entangle_params[i], i, i + 1)

        circ.measure_all()
        return circ

    def run(self, data: np.ndarray, rotation_params: np.ndarray, entangle_params: np.ndarray, shots: int = None) -> float:
        """
        Run the quantum attention circuit on a single 2‑D image patch.

        Args:
            data: 2‑D array of shape (kernel_size, kernel_size) with integer pixel values.
            rotation_params: 1‑D array of length n_qubits, used for ry gates.
            entangle_params: 1‑D array of length n_qubits‑1, used for crx gates.
            shots: number of measurement shots (defaults to self.shots).

        Returns:
            Float: average probability of measuring |1> across all qubits.
        """
        shots = shots or self.shots

        # Bind the data to the rotation parameters
        param_binds = []
        flat = data.reshape(1, self.n_qubits)
        for vec in flat:
            bind = {self.theta[i]: np.pi if vec[i] > self.threshold else 0 for i in range(self.n_qubits)}
            param_binds.append(bind)

        # Build the full circuit
        circ = self._build_attention(rotation_params, entangle_params)

        # Execute
        job = execute(circ, self.backend, shots=shots, parameter_binds=param_binds)
        result = job.result().get_counts(circ)

        # Compute average |1> probability
        counts = 0
        for key, val in result.items():
            ones = sum(int(bit) for bit in key)
            counts += ones * val

        return counts / (shots * self.n_qubits)

__all__ = ["SelfAttention"]
