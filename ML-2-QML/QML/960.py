"""HybridEstimator implementation for quantum neural networks.

This module defines a variational quantum circuit that can be trained
end‑to‑end using Pennylane.  The circuit supports multiple rotation
layers, entanglement via CNOTs, and configurable input encoding.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from typing import Sequence

class HybridEstimator:
    """A variational quantum neural network.

    Parameters
    ----------
    input_dim : int
        Number of classical input features.
    num_layers : int, optional
        Number of parameterized rotation layers.  Defaults to 2.
    wires : int | None, optional
        Number of qubits to use.  If None, ``input_dim`` is used.
    device : str | qml.Device, optional
        Pennylane device to run the circuit on.  Defaults to
        ``default.qubit`` with ``wires`` wires.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 2,
        wires: int | None = None,
        device: str | qml.Device | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.wires = wires or input_dim
        self.device = device or qml.device("default.qubit", wires=self.wires)
        # Initialize trainable weights: one rotation per layer per wire
        self.weights = np.random.randn(self.num_layers, self.wires, 3)

    def _circuit(self, inputs: Sequence[float], weights: np.ndarray) -> float:
        """Variational circuit returning the expectation of Z on the first qubit."""
        for i, wire in enumerate(range(self.wires)):
            qml.RY(inputs[i], wires=wire)
        for layer in range(self.num_layers):
            for wire in range(self.wires):
                theta, phi, lam = weights[layer, wire]
                qml.Rot(theta, phi, lam, wires=wire)
            # Entangle all qubits in a ring
            for wire in range(self.wires):
                qml.CNOT(wires=[wire, (wire + 1) % self.wires])
        return qml.expval(qml.PauliZ(0))

    def __call__(self, inputs: Sequence[float]) -> float:
        """Return the circuit output for a single input vector."""
        return qml.execute(
            [self._circuit],
            self.device,
            gradient_fn=None,
            args=[inputs, self.weights],
        )[0]

    def loss(self, inputs: Sequence[float], target: float) -> float:
        """Mean‑squared‑error loss for a single data point."""
        pred = self.__call__(inputs)
        return (pred - target) ** 2

    def train(
        self,
        dataset: Sequence[tuple[Sequence[float], float]],
        epochs: int = 100,
        lr: float = 0.01,
    ) -> None:
        """Simple gradient‑descent training loop."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            for x, y in dataset:
                grads = opt.compute_gradient(lambda w: self.loss(x, y), self.weights)
                self.weights = opt.apply_gradients(self.weights, grads)

__all__ = ["HybridEstimator"]
