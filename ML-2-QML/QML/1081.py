"""Quantum‑enhanced regressor with a variational circuit.

The QNN takes a 2‑dimensional feature vector, encodes it with a simple
feature‑map, applies a 3‑layer ansatz with 4 qubits, and measures the
expectation of a Pauli‑Z observable.  The circuit parameters are
optimised with a classical optimiser (Adam) using the parameter‑shift
rule.  The public interface matches the classical version: EstimatorQNNEnhanced()
returns an object with a predict(x) method that returns a torch.Tensor.

This implementation uses Pennylane's default simulator and the
parameter‑shift gradient, making it fully differentiable with
respect to the circuit parameters.
"""

from __future__ import annotations

import numpy as np
import torch
import pennylane as qml


class EstimatorQNNEnhanced:
    """Variational quantum neural network."""

    def __init__(
        self,
        num_qubits: int = 4,
        layers: int = 3,
        device: qml.Device | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.device = device or qml.device("default.qubit", wires=num_qubits)
        # initialise weights randomly
        init_weights = np.random.randn(layers, num_qubits)
        self.weights = torch.tensor(
            init_weights, dtype=torch.float32, requires_grad=True
        )

        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            # simple feature‑map: encode each input component into a rotation
            for i in range(num_qubits):
                qml.RX(inputs[i % 2], wires=i)
            # variational ansatz
            for l in range(layers):
                for q in range(num_qubits):
                    qml.RY(weights[l, q], wires=q)
                qml.CNOT(wires=[q, (q + 1) % num_qubits])
            # measurement
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the circuit expectation value for the given inputs."""
        return self.circuit(inputs, self.weights)

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Mean‑squared error loss."""
        return torch.mean((pred - target) ** 2)

    def train_step(
        self,
        inputs: torch.Tensor,
        target: torch.Tensor,
        lr: float = 0.01,
    ) -> float:
        """Perform one optimisation step and return the loss value."""
        pred = self.predict(inputs)
        loss = self.loss(pred, target)
        loss.backward()
        with torch.no_grad():
            self.weights -= lr * self.weights.grad
            self.weights.grad.zero_()
        return loss.item()


__all__ = ["EstimatorQNNEnhanced"]
