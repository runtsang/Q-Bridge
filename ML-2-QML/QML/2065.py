"""Quantum‑enhanced convolution using a variational circuit with gradient access."""

import pennylane as qml
import numpy as np

class ConvEnhanced:
    """
    Variational quantum convolutional filter. The filter operates on a 2‑D kernel
    reshaped to a 1‑D array of qubit amplitudes. The circuit contains a trainable
    rotation gate on each qubit followed by a layer of CNOTs for entanglement.
    The output is the mean expectation value of Z on all qubits and can be
    differentiated with PennyLane's autograd interface.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 1000, device: str | None = None):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.shots = shots
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=self.shots)
        self.params = np.random.uniform(0, 2 * np.pi, (self.n_qubits,))

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, params):
            # Encode data into rotation angles (RY)
            for i, val in enumerate(x):
                qml.RY(val * np.pi, wires=i)
            # Variational layer
            for i in range(self.n_qubits):
                qml.RZ(params[i], wires=i)
            # Simple entanglement pattern
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of Z on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """
        Evaluate the quantum filter on a 2‑D kernel.

        Parameters
        ----------
        data : array_like
            2‑D array with shape (kernel_size, kernel_size). Values are assumed
            to be in the range [0, 1] and are mapped to rotation angles.

        Returns
        -------
        float
            Mean expectation value of Pauli‑Z across all qubits.
        """
        x = data.reshape(-1)  # flatten
        expvals = self.circuit(x, self.params)
        return float(np.mean(expvals))

    def train_step(self, data: np.ndarray, target: float, lr: float = 0.01):
        """
        One gradient‑based update step using the autograd interface.

        Parameters
        ----------
        data : array_like
            Input kernel.
        target : float
            Desired output value.
        lr : float
            Learning rate.
        """
        def loss_fn(params):
            return (self.circuit(data.reshape(-1), params) - target) ** 2

        grads = qml.grad(loss_fn)(self.params)
        self.params -= lr * grads

    def set_params(self, params: np.ndarray):
        """
        Replace the trainable parameters.
        """
        self.params = params

__all__ = ["ConvEnhanced"]
