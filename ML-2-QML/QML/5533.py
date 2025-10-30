import numpy as np
from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, Parameter
from typing import Iterable

class HybridFCL:
    """
    Quantum implementation of the hybrid fully connected layer.
    The circuit consists of a linear encoder (classical) that maps
    the input into a single parameter, a parameterised Ry rotation
    applied to each qubit, and a measurement in the Z basis.
    The expectation value of the Pauli‑Z observable is returned
    as the layer output, followed by a classical linear head.
    """

    def __init__(self, n_features: int = 1, n_qubits: int = 1, shots: int = 100) -> None:
        self.n_features = n_features
        self.n_qubits = n_qubits
        self.shots = shots

        # Classical linear encoder (trainable via numpy arrays for simplicity)
        self.encoder_weights = np.random.randn(n_features, 1).astype(np.float64)
        self.encoder_bias = np.random.randn(1).astype(np.float64)

        # Build the quantum circuit
        self.theta = Parameter("θ")
        self._qc = QuantumCircuit(n_qubits)
        self._qc.h(range(n_qubits))
        self._qc.ry(self.theta, range(n_qubits))
        self._qc.measure_all()

        # Backend
        self.backend = Aer.get_backend("qasm_simulator")

        # Classical head for final prediction
        self.head_weights = np.random.randn(1, 1).astype(np.float64)
        self.head_bias = np.random.randn(1).astype(np.float64)

    def _encode(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Project the raw theta values into a single scalar per sample
        using the classical encoder.  Returns a 1‑D NumPy array.
        """
        theta_arr = np.array(list(thetas), dtype=np.float64).reshape(-1, 1)
        encoded = theta_arr @ self.encoder_weights + self.encoder_bias
        return encoded.squeeze(-1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Execute the quantum circuit for each encoded theta and
        return the expectation value of the Pauli‑Z operator
        after applying the classical head.
        """
        encoded = self._encode(thetas)
        job = execute(
            self._qc,
            self.backend,
            shots=self.shots,
            parameter_binds=[{self.theta: val} for val in encoded]
        )
        result = job.result()
        counts = result.get_counts(self._qc)
        probs = np.array(list(counts.values())) / self.shots
        states = np.array([int(k, 2) for k in counts.keys()])
        # Convert bitstring to +/-1 eigenvalues of Z (single qubit)
        z_vals = 1 - 2 * states
        expectation = np.sum(z_vals * probs)
        # Apply classical linear head
        output = expectation * self.head_weights.item() + self.head_bias.item()
        return np.array([output])

__all__ = ["HybridFCL"]
