"""ConvEnhancedQ – quantum‑enhanced convolutional filter using a parameter‑shared variational circuit."""

import pennylane as qml
import pennylane.numpy as np
from typing import Tuple

class ConvEnhancedQ:
    """
    Quantum implementation of ConvEnhanced. Uses a parameter‑shared
    variational circuit to emulate a depthwise separable convolution.
    """

    def __init__(self, kernel_size: int = 2, shots: int = 1024, device_name: str = "default.qubit"):
        self.kernel_size = kernel_size
        self.n_qubits = kernel_size ** 2
        self.device = qml.device(device_name, wires=self.n_qubits, shots=shots)
        self.params = np.random.uniform(0, 2 * np.pi, self.n_qubits)

        @qml.qnode(self.device, interface="autograd")
        def circuit(*param_values):
            for i in range(self.n_qubits):
                qml.RX(param_values[i], wires=i)
            # Simple entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            qml.CNOT(wires=[self.n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """
        Run the variational circuit on flattened data.

        Parameters
        ----------
        data : array-like, shape (kernel_size, kernel_size)
            Input pixel values.

        Returns
        -------
        float
            Expectation value of PauliZ on the first qubit.
        """
        flat = np.reshape(data, self.n_qubits)
        # Map pixel values to parameters via a simple linear mapping
        param_values = self.params * flat
        return float(self.circuit(*param_values))

__all__ = ["ConvEnhancedQ"]
