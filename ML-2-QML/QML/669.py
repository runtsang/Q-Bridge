"""Quantum-enhanced quanvolution network using Pennylane."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class Quanvolution__gen329(nn.Module):
    """
    Quantum quanvolution network that replaces the random layer with a trainable variational
    circuit and uses Pennylane's autograd to back‑propagate through the quantum part.
    """
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.device = qml.device("default.qubit", wires=self.n_wires)
        self.num_layers = 3
        # parameters for the variational circuit
        self.params = nn.Parameter(torch.randn(self.num_layers, self.n_wires, 3))
        # define the qnode
        @qml.qnode(self.device, interface="torch")
        def _circuit(data: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            """
            data: shape (batch, 4) or (4,)
            params: shape (num_layers, n_wires, 3)
            """
            # encode data via Ry rotations
            for i in range(self.n_wires):
                qml.RY(data[..., i], wires=i)
            # variational layers
            qml.templates.StronglyEntanglingLayers(params, wires=range(self.n_wires))
            # measure expectation of Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        self.circuit = _circuit
        self.linear = nn.Linear(self.n_wires * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, 28, 28)
        batch_size = x.size(0)
        # extract 2x2 patches
        patches = x.unfold(2, 2, 2).unfold(3, 2, 2)  # (batch, 1, 14, 14, 2, 2)
        patches = patches.squeeze(1)  # (batch, 14, 14, 2, 2)
        patches = patches.reshape(batch_size, 14 * 14, 4)  # (batch, 196, 4)
        # apply quantum circuit to each patch
        q_features = []
        for i in range(patches.size(1)):
            data = patches[:, i, :]  # (batch, 4)
            q_output = self.circuit(data, self.params)  # (batch, 4)
            q_features.append(q_output)
        q_features = torch.stack(q_features, dim=1)  # (batch, 196, 4)
        q_features = q_features.reshape(batch_size, -1)  # (batch, 196*4)
        logits = self.linear(q_features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["Quanvolution__gen329"]
