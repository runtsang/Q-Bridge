"""Variational quanvolution filter using Pennylane."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

class QuanvolutionFilter(nn.Module):
    """Two‑by‑two quantum kernel with a learnable variational circuit."""

    def __init__(self, n_qubits: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.param_shape = (n_qubits * n_layers * 3,)
        self.params = nn.Parameter(torch.randn(self.param_shape))
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(patch: torch.Tensor) -> torch.Tensor:
            # Encode the 4‑pixel patch into rotation angles
            for i in range(self.n_qubits):
                qml.RY(patch[i], wires=i)
            offset = 0
            # Variational layers
            for _ in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(self.params[offset], wires=i); offset += 1
                    qml.RZ(self.params[offset], wires=i); offset += 1
                    qml.RY(self.params[offset], wires=i); offset += 1
                # Entanglement chain
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r:r + 2, c:c + 2].view(bsz, 4)
                # Scale pixel values to [0, π] for rotation angles
                patch_scaled = patch * torch.pi
                # Compute quantum features for each batch element
                out = torch.stack([self.circuit(patch_scaled[i]) for i in range(bsz)])
                patches.append(out)
        return torch.cat(patches, dim=1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that uses the variational quanvolution filter."""

    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
