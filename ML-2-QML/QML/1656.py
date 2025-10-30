"""Quantum version of ConvEnhanced using Pennylane."""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Tuple, Optional
from pennylane import numpy as pnp


class ConvEnhancedQML:
    """
    QML implementation of the ConvEnhanced filter.

    Parameters
    ----------
    in_channels : int
        Number of input feature maps.
    out_channels : int
        Number of output feature maps.
    kernel_size : int or tuple[int, int] = 2
        Size of the convolution kernel (square by default).
    depthwise : bool = False
        If True, apply depthwise separable strategy (one circuit per input channel).
    threshold : float | None = None
        Learnable threshold applied after sigmoid. If None, a fixed value of 0.0 is used.
    device : str = "default.qubit"
        PennyLane device name.
    shots : int = 1024
        Number of measurement shots.
    """
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int] = 2,
        depthwise: bool = False,
        threshold: Optional[float] = None,
        device: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.depthwise = depthwise

        n_qubits = kernel_size[0] * kernel_size[1]
        self.dev = qml.device(device, wires=n_qubits, shots=shots)

        # Parameter vector for the ansatz
        self.params = pnp.random.rand(n_qubits * 2)

        # Learnable threshold
        self.threshold = pnp.array(threshold if threshold is not None else 0.0)

    def _ansatz(self, x, params):
        """Simple variational ansatz: RY rotations + CNOT ladder."""
        for i in range(len(x)):
            qml.RY(x[i], wires=i)
        for i in range(len(x)):
            qml.RZ(params[i], wires=i)
        for i in range(len(x) - 1):
            qml.CNOT(wires=[i, i + 1])

    @qml.qnode
    def _qnode(self, x, params):
        self._ansatz(x, params)
        return [qml.expval(qml.PauliZ(i)) for i in range(len(x))]

    def run(self, data: np.ndarray | Iterable) -> float:
        """
        Execute the quantum circuit on classical data.

        Parameters
        ----------
        data : np.ndarray
            2D array with shape (kernel_size, kernel_size) or (B, C, H, W).

        Returns
        -------
        float
            Mean probability of measuring |1> after sigmoid and threshold.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        if data.ndim == 2:
            # Single sample, single channel
            data = data.reshape(1, 1, *self.kernel_size)

        if data.ndim == 4 and data.shape[0] == 1:
            data = data.squeeze(0)  # (C, H, W)

        # Normalize data to [0, π] for RX encoding
        x = data.astype(np.float32).reshape(-1) * np.pi

        expvals = self._qnode(x, self.params)
        probs = 0.5 * (1 - np.array(expvals))  # map Z expectation to |0> probability
        probs = np.clip(probs, 0, 1)

        activations = 1.0 / (1.0 + np.exp(-(probs - self.threshold)))
        return activations.mean()

    def train(self, data: np.ndarray, labels: np.ndarray, lr: float = 0.01, epochs: int = 10):
        """
        Simple gradient‑descent training loop for the variational parameters.

        Parameters
        ----------
        data : np.ndarray
            Input samples of shape (N, kernel_size, kernel_size).
        labels : np.ndarray
            Binary labels of shape (N,).
        lr : float
            Learning rate.
        epochs : int
            Number of training epochs.
        """
        optimizer = qml.GradientDescentOptimizer(lr)

        for epoch in range(epochs):
            loss = 0.0
            for x, y in zip(data, labels):
                def loss_fn(p):
                    preds = self._qnode(x.reshape(-1), p)
                    probs = 0.5 * (1 - np.array(preds))
                    probs = np.clip(probs, 0, 1)
                    logits = probs - self.threshold
                    pred = 1.0 / (1.0 + np.exp(-logits))
                    return (pred - y) ** 2

                self.params = optimizer.step(loss_fn, self.params)
                loss += loss_fn(self.params)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss / len(data):.4f}")

    def __call__(self, data) -> float:
        return self.run(data)


__all__ = ["ConvEnhancedQML"]
