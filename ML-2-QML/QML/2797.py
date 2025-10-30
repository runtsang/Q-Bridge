"""Quantum hybrid self‑attention with quanvolutional encoding."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq


class HybridSelfAttentionQuanvolution(tq.QuantumModule):
    """Quantum module that encodes 2×2 image patches, applies a random layer, and computes attention via expectation‑value correlations."""

    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
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
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))

        features = torch.cat(patches, dim=1)  # (batch, 4*14*14)

        # Compute attention via pairwise cosine‑like similarity of expectation values
        queries = features @ features.T
        scores = torch.softmax(queries / np.sqrt(features.shape[-1]), dim=-1)
        attn_features = scores @ features

        logits = self.linear(attn_features)
        return F.log_softmax(logits, dim=-1)


__all__ = ["HybridSelfAttentionQuanvolution"]
