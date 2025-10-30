"""Quanvolutional filter using a trainable variational quantum circuit and attention."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuanvolutionHybrid(nn.Module):
    """
    Quantum hybrid network that replaces the random layer with a
    parameterized variational circuit.  After the quantum filter, a
    patch‑wise attention mechanism is applied before the linear head.
    """

    def __init__(self,
                 in_channels: int = 1,
                 hidden_dim: int = 64,
                 num_classes: int = 10,
                 device: str = "default.qubit") -> None:
        super().__init__()
        self.num_wires = 4
        self.dev = qml.device(device, wires=self.num_wires)
        # Trainable parameters for a two‑layer variational circuit
        self.variational_params = nn.Parameter(torch.randn(2, self.num_wires))
        # Attention head
        self.attention = nn.Sequential(
            nn.Linear(self.num_wires, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.linear = nn.Linear(self.num_wires * 14 * 14, num_classes)

    def quantum_layer(self, data: torch.Tensor) -> torch.Tensor:
        """
        Run a variational circuit on a batch of 4‑dim data.
        Returns a tensor of shape (batch, 4) containing expectation values.
        """
        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            # Encode classical data into rotations
            for i in range(self.num_wires):
                qml.RY(x[i], wires=i)
            # Variational layers
            for layer in range(params.shape[0]):
                for i in range(self.num_wires):
                    qml.RY(params[layer, i], wires=i)
                # Entangling pattern
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
            # Measure expectation of Pauli‑Z on each wire
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_wires)]

        batch = data.shape[0]
        results = []
        for i in range(batch):
            out = circuit(data[i], self.variational_params)
            results.append(out)
        return torch.stack(results)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )  # (B, 4)
                measurement = self.quantum_layer(data)  # (B, 4)
                patches.append(measurement)
        patches_tensor = torch.stack(patches, dim=1)  # (B, 14*14, 4)
        # Attention over patches
        attn_scores = self.attention(patches_tensor).squeeze(-1)  # (B, 14*14)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, 14*14, 1)
        weighted = patches_tensor * attn_weights  # (B, 14*14, 4)
        flat = weighted.reshape(x.size(0), -1)  # (B, 4*14*14)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
