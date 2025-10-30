"""Quantum quanvolution classifier using PennyLane.

The model slices each 28×28 image into 2×2 patches, encodes the
pixel intensities with RY rotations, and applies a small variational
circuit. The expectation values of Z on each qubit form a 4‑dim
feature per patch. A shared linear head maps the concatenated
features to class logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


class Quanvolution__gen124(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.n_wires = 4
        self.device = qml.device("default.qubit", wires=self.n_wires)

        # Trainable parameters for the variational layer
        self.params = nn.Parameter(torch.randn(self.n_wires))

        # Linear head
        self.classifier = nn.Linear(4 * 14 * 14, num_classes)

        @qml.qnode(self.device, interface="torch")
        def circuit(inputs: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
            # Encode input data
            for i in range(self.n_wires):
                qml.RY(inputs[i], wires=i)
            # Variational layer
            for i in range(self.n_wires):
                qml.RY(params[i], wires=i)
            # Entangling layer
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[2, 3])
            # Measure expectation values of PauliZ
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x shape: (batch, 1, 28, 28)
        batch_size = x.size(0)
        features = []
        for b in range(batch_size):
            img = x[b, 0]  # (28, 28)
            patch_features = []
            for r in range(0, 28, 2):
                for c in range(0, 28, 2):
                    patch = img[r:r + 2, c:c + 2].reshape(4)
                    qfeat = self.circuit(patch, self.params)
                    patch_features.append(qfeat)
            patch_features = torch.cat(patch_features, dim=0)  # (4*14*14)
            features.append(patch_features)
        features = torch.stack(features, dim=0)  # (batch, 4*14*14)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["Quanvolution__gen124"]
