"""Quantum convolution filter implemented with PennyLane.

The class ``ConvFilter`` mimics the interface of the classical version
and can be used as a drop‑in replacement.  It uses a parameterised
``StronglyEntanglingLayers`` ansatz that is differentiable via PennyLane’s
autograd interface.  The input data is encoded by rotating each qubit with an
angle proportional to the pixel value; the circuit returns the average
probability of measuring ``|1⟩`` over all qubits.

The public API mirrors the original seed’s ``Conv`` factory:
``Conv()`` returns an instance of ``ConvFilter``.
"""

import pennylane as qml
import numpy as np
from typing import Tuple


class ConvFilter:
    """Variational quantum convolution filter.

    Parameters
    ----------
    kernel_size : int, default 3
        Size of the square kernel (qubit count = kernel_size**2).
    threshold : float, default 0.5
        Pixel threshold used for binary encoding of the data.
    device : str, default "default.qubit"
        PennyLane device to run the circuit on.
    shots : int, default 1024
        Number of shots for the simulator.
    n_layers : int, default 2
        Number of entangling layers in the ansatz.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        threshold: float = 0.5,
        device: str = "default.qubit",
        shots: int = 1024,
        n_layers: int = 2,
    ) -> None:
        self.n_qubits = kernel_size ** 2
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_layers = n_layers

        self.dev = qml.device(device, wires=self.n_qubits, shots=shots)

        # Trainable parameters for each layer and qubit
        self.params = np.random.normal(
            0, np.pi / 2, size=(self.n_layers, self.n_qubits)
        )

        # Compile the circuit once
        self._circuit = qml.QNode(self._qcircuit, self.dev, interface="autograd")

    def _qcircuit(self, params: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Variational circuit that encodes data and applies the ansatz."""
        # Data re‑upload: rotate each qubit by π * pixel_value
        for i in range(self.n_qubits):
            qml.RX(data[i] * np.pi, wires=i)

        # Entangling layers
        qml.templates.StronglyEntanglingLayers(params, wires=range(self.n_qubits))

        # Return probabilities for all basis states
        return qml.probs(wires=range(self.n_qubits))

    def run(self, data: np.ndarray) -> float:
        """Run the circuit on a single kernel-sized patch.

        Parameters
        ----------
        data : array-like
            2‑D array of shape ``(kernel_size, kernel_size)``.

        Returns
        -------
        float
            Average probability of measuring ``|1⟩`` across all qubits.
        """
        flat = np.asarray(data).reshape(-1)
        probs = self._circuit(self.params, flat)
        # Probabilities for |1⟩ states are at odd indices
        return probs[1::2].mean()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply the quantum filter to a batch of images.

        Parameters
        ----------
        x : array-like
            Input array of shape ``(batch, channels, height, width)``.

        Returns
        -------
        np.ndarray
            Output array after filtering.
        """
        batch, channels, H, W = x.shape
        stride = 1
        out_h = (H - self.kernel_size) // stride + 1
        out_w = (W - self.kernel_size) // stride + 1
        out = np.zeros((batch, channels, out_h, out_w))

        for b in range(batch):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        patch = x[b, c, i : i + self.kernel_size, j : j + self.kernel_size]
                        out[b, c, i, j] = self.run(patch)

        return out


def Conv() -> ConvFilter:
    """Factory that returns a ``ConvFilter`` instance with default parameters."""
    return ConvFilter()


__all__ = ["ConvFilter", "Conv"]
