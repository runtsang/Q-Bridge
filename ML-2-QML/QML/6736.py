"""Quantum-enhanced quanvolution module using a trainable variational circuit."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class VariationalQuantumBlock(tq.QuantumModule):
    """Parameter‑driven 4‑qubit variational circuit with two layers."""

    def __init__(self, n_wires: int = 4, n_layers: int = 2):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.params = nn.Parameter(torch.randn(n_layers, n_wires))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)

        # Encode classical patch
        self.encoder(qdev, x)

        # Variational layers
        for i in range(self.n_layers):
            for w in range(self.n_wires):
                qdev.rz(self.params[i, w], wires=w)
            # Entangle neighboring qubits
            for w in range(self.n_wires - 1):
                qdev.cx(wires=[w, w + 1])

        measurement = self.measure(qdev)
        return measurement.view(bsz, self.n_wires)


class QuanvolutionHybrid(nn.Module):
    """Hybrid network that applies a trainable quantum filter to image patches."""

    def __init__(self, n_wires: int = 4, n_layers: int = 2, num_classes: int = 10):
        super().__init__()
        self.quantum_block = VariationalQuantumBlock(n_wires, n_layers)
        self.linear = nn.Linear(n_wires * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        x = x.view(bsz, 28, 28)

        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = x[:, r : r + 2, c : c + 2]  # (B, 2, 2)
                patch = patch.reshape(bsz, 4)  # (B, 4)
                out = self.quantum_block(patch)
                patches.append(out)

        features = torch.cat(patches, dim=1)  # (B, 4*14*14)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
