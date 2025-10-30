"""Quantum quanvolution filter with a trainable variational circuit.

This module replaces the RandomLayer of the seed with a fully differentiable
variational layer (tq.QuantumLayer).  The encoder applies Ry rotations
parameterised by the pixel values.  After the variational layer, all qubits are
measured in the Pauli‑Z basis.  The classifier mirrors the classical API
returning both log‑softmax logits and an embedding tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class QuanvolutionFilter(tq.QuantumModule):
    """Variational 2‑qubit filter operating on 2×2 patches."""

    def __init__(self, n_wires: int = 4, n_layers: int = 3) -> None:
        super().__init__()
        self.n_wires = n_wires
        # Encoder: encode pixel intensities into Ry rotations
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Variational circuit with n_layers of trainable gates
        circuit = tq.Circuit()
        for _ in range(n_layers):
            circuit += tq.RX(0.0, wires=0)
            circuit += tq.RY(0.0, wires=1)
            circuit += tq.RZ(0.0, wires=2)
            circuit += tq.H(3)
            circuit += tq.CNOT(0, 1)
            circuit += tq.CNOT(1, 2)
            circuit += tq.CNOT(2, 3)
        self.var_layer = tq.QuantumLayer(
            circuit, n_wires=self.n_wires, n_params=self.n_wires * 3 * n_layers
        )
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
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
                )
                self.encoder(qdev, data)
                self.var_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)


class QuanvolutionClassifier(nn.Module):
    """Hybrid quantum‑classical classifier with an embedding head.

    The forward method returns a dictionary with ``logits`` and ``embed``,
    matching the API of the classical counterpart.
    """

    def __init__(self, num_classes: int = 10, embed_dim: int = 128) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fc = nn.Linear(4 * 14 * 14, embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        quantum_features = self.qfilter(x)      # (B, 784)
        embed = F.relu(self.fc(quantum_features))
        logits = self.cls_head(embed)
        return {
            "logits": F.log_softmax(logits, dim=-1),
            "embed": embed,
        }


__all__ = ["QuanvolutionFilter", "QuanvolutionClassifier"]
