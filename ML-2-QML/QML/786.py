"""Quantum‑enhanced convolution using Pennylane with trainable angles."""

import numpy as np
import pennylane as qml


class ConvQuantum:
    """
    Variational quantum convolution filter that
    **directly** satisfies the classical convolution
    *differentiable* with respect to all parameters.
    The quantum circuit uses a 2×2‑kernel (i.e., 4 qubits) and
    **all θ** tuned by ~ simulated GPU‑accelerated QPU.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.5,
        device: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.device = qml.device(device, wires=self.n_qubits, shots=shots)
        # Initialize trainable parameters for the variational layer
        self.variational_params = np.random.uniform(0, 2 * np.pi, self.n_qubits)

        @qml.qnode(self.device, interface="autograd")
        def circuit(data):
            # Encode data into RX rotations
            for i in range(self.n_qubits):
                angle = np.pi if data[i] > self.threshold else 0.0
                qml.RX(angle, wires=i)
            # Variational layer
            for i in range(self.n_qubits):
                qml.RY(self.variational_params[i], wires=i)
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Expectation of PauliZ on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data):
        """
        Run the quantum circuit on classical data.

        Args:
            data: 2D array with shape (kernel_size, kernel_size).

        Returns:
            float: average probability of measuring |1> across qubits.
        """
        data = np.asarray(data).reshape(-1)
        expvals = self.circuit(data)
        probs = [(1 - e) / 2 for e in expvals]
        return float(np.mean(probs))


__all__ = ["ConvQuantum"]
