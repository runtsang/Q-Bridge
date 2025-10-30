"""Quantum convolution filter using a variational circuit.

The public API mirrors the original: Conv() returns a callable object with a run(data) method.
"""

import pennylane as qml
import torch
import numpy as np

class ConvEnhanced(torch.nn.Module):
    """
    Variational quantum convolution filter.

    Parameters
    ----------
    kernel_size : int
        Size of the square input patch (e.g., 2 for a 2×2 filter).
    threshold : float
        Pixel threshold used for angle encoding.
    n_layers : int
        Number of variational layers.
    device : str | None
        PennyLane device name; defaults to 'default.qubit'.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.5,
        n_layers: int = 1,
        device: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.n_layers = n_layers
        self.dev = qml.device(device or "default.qubit", wires=self.n_qubits)

        # Variational parameters
        self.weights = torch.nn.Parameter(
            0.01 * torch.randn(n_layers, self.n_qubits, dtype=torch.float32)
        )

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs, weights):
            # Angle encoding: rotate each qubit by π if pixel > threshold
            for i, val in enumerate(inputs):
                angle = torch.where(val > self.threshold, torch.tensor(np.pi), torch.tensor(0.0))
                qml.RY(angle, wires=i)

            # Variational layers
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                # Entanglement pattern
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            # Expectation value of PauliZ on first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, data):
        """
        Compute the average probability of measuring |1> across all qubits.

        Parameters
        ----------
        data : array‑like
            2‑D array of shape (kernel_size, kernel_size).

        Returns
        -------
        float
            Scalar activation.
        """
        flat = torch.tensor(data, dtype=torch.float32).flatten()
        out = self.circuit(flat, self.weights)
        # Convert expectation value to probability of |1>
        prob = (1 - out) / 2
        return prob.mean().item()

    def run(self, data):
        """Compatibility wrapper for the original API."""
        return self.forward(data)

def Conv(**kwargs):
    """
    Factory function to create a ConvEnhanced instance.

    Accepts the same keyword arguments as ConvEnhanced.
    """
    return ConvEnhanced(**kwargs)
