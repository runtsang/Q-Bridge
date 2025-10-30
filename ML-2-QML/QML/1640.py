"""Quantum‑inspired quanvolution model using PennyLane.

The QML variant replaces the classical convolution with a variational
quantum kernel that processes each 2×2 patch.  A single trainable
parameter vector is shared across all patches, enabling efficient
gradient propagation while still capturing non‑linear features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class QuanvolutionHybrid(nn.Module):
    """Hybrid quantum‑classical quanvolution."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.num_patches = 14 * 14
        self.dev = qml.device("default.qubit", wires=self.n_wires)
        self.vqc_params = nn.Parameter(torch.randn(self.n_wires))
        self.fc = nn.Linear(4 * self.num_patches, 10)
        self.bias = nn.Parameter(torch.zeros(10))

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode 4‑element patch into qubit rotations
            for i in range(self.n_wires):
                qml.RY(inputs[i], wires=i)
            # Entanglement pattern
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])
            # Variational layer
            for i in range(self.n_wires):
                qml.RY(params[i], wires=i)
            # Return expectation values of Pauli‑Z
            return (
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliZ(1)),
                qml.expval(qml.PauliZ(2)),
                qml.expval(qml.PauliZ(3)),
            )

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        q_features = []

        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = (
                    x[:, 0, r : r + 2, c : c + 2]
                   .reshape(batch_size, -1)
                )
                out = self.circuit(patch, self.vqc_params)
                out = torch.stack(out, dim=1)
                q_features.append(out)

        q_vector = torch.cat(q_features, dim=1).reshape(batch_size, -1)
        logits = self.fc(q_vector)
        logits = logits - self.bias
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
