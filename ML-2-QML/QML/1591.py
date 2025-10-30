"""Hybrid quanvolution network with variational quantum kernel."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuanvolutionNet(tq.QuantumModule):
    """Hybrid network that applies a parametric variational circuit to image patches.
    The circuit consists of a sequence of two‑qubit gates trained jointly with the
    classical linear head.  Supports multi‑task classification via multiple heads.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 10, heads: int = 1):
        super().__init__()
        self.n_wires = 4  # 2x2 patches → 4 qubits
        # Simple Ry encoder for each pixel intensity
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        # Trainable variational layer of 8 two‑qubit operations
        self.var_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)), init="random")
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Linear heads for multi‑task classification
        self.heads = nn.ModuleList([nn.Linear(4 * 14 * 14, num_classes) for _ in range(heads)])

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
        features = torch.cat(patches, dim=1)
        logits = [F.log_softmax(head(features), dim=-1) for head in self.heads]
        return torch.stack(logits, dim=1) if len(self.heads) > 1 else logits[0]
