"""Quantum neural network with a hybrid classical post‑processing head."""

from __future__ import annotations

import pennylane as qml
import torch
from pennylane import numpy as np

__all__ = ["QCNNModel"]


class QCNNModel:
    """
    A hybrid QCNN that maps an 8‑dimensional classical vector to a probability.

    Architecture:
        * Feature map: Multi‑layer AngleEmbedding (3 layers of 8 rotations + 8 CNOTs).
        * Ansatz: Strongly entangling layers (2 layers, 8 wires) with trainable rotations.
        * Measurement: Expectation value of PauliZ on wire 0, converted to a probability via a sigmoid.

    The class exposes a :py:meth:`forward` method that can be used with PyTorch autograd
    by wrapping the QNode with :class:`pennylane.QNode`.  This allows end‑to‑end
    optimisation of both classical and quantum parameters.
    """

    def __init__(
        self,
        device: qml.Device | None = None,
        n_layers: int = 2,
        n_wires: int = 8,
    ) -> None:
        if device is None:
            device = qml.device("default.qubit", wires=n_wires)
        self.device = device
        self.n_wires = n_wires

        # Create the QNode
        @qml.qnode(self.device, interface="torch", diff_method="parameter-shift")
        def circuit(x, weights):
            # Feature map: 3 layers of AngleEmbedding + CNOTs
            for _ in range(3):
                qml.templates.AngleEmbedding(x, wires=range(n_wires))
                qml.templates.StronglyEntanglingLayers(weights, wires=range(n_wires))
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

        # Initialise ansatz parameters
        self.weights = torch.randn(n_layers * n_wires * 3, requires_grad=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning a probability.

        Parameters
        ----------
        inputs : torch.Tensor
            1‑D tensor of shape (n_wires,) representing the classical input.

        Returns
        -------
        torch.Tensor
            A scalar tensor in (0, 1) representing the predicted probability.
        """
        # Ensure inputs are torch tensors of correct shape
        if inputs.ndim!= 1 or inputs.shape[0]!= self.n_wires:
            raise ValueError(f"Expected input of shape ({{self.n_wires}},), got {inputs.shape}")
        # Run the QNode
        expectation = self.circuit(inputs, self.weights)
        # Map expectation from [-1, 1] to [0, 1] via sigmoid
        return torch.sigmoid(expectation)
