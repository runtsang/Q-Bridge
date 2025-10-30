"""Quantum‑convolution filter with trainable parameters and hybrid loss."""
import numpy as np
import pennylane as qml

class ConvFilter:
    """
    Quantum filter that replaces the classical Conv.
    Implements a 2×2 filter as a quantum circuit with a
    parameter‑shaped circuit (RX‑RX‑CNOT) and a metric
    that can be gradient‑free or hybrid‑autograd.
    """

    def __init__(self, kernel_size: int = 2, backend: str = "default.qubit", shots: int = 1000, threshold: float = 0.5):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.backend = backend
        self.shots = shots
        self.threshold = threshold

        # Construct a variational circuit with a rotation‑only
        # parameterized gate pattern that keeps the quantum
        # complexity low while still fully trainable.
        self.dev = qml.device(backend, wires=self.n_qubits)
        self.theta = np.random.uniform(0, 2 * np.pi, self.n_qubits)
        self.circuit = self._make_circuit()

    def _make_circuit(self):
        """Return a PennyLane QNode that uses a trainable circuit."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(data, theta):
            # data: flattened input values mapped to [0, pi] via threshold
            for i in range(self.n_qubits):
                qml.RX(data[i], wires=i)
                qml.RX(theta[i], wires=i)
            # Entanglement pattern
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement: average probability of measuring |1> across qubits
            expvals = [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
            probs = [(1 - e) / 2 for e in expvals]
            return sum(probs) / self.n_qubits
        return circuit

    def run(self, data):
        """
        Run the quantum circuit on classical data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        # Map data values to [0, pi] based on threshold
        data_flat = data.reshape(-1)
        data_mapped = np.where(data_flat > self.threshold, np.pi, 0.0)

        # Execute the circuit
        result = self.circuit(data_mapped, self.theta)

        return float(result)

__all__ = ["ConvFilter"]
