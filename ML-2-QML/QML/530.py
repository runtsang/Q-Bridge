"""
Hybrid Quanvolution network with a parameterized quantum kernel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from torchquantum import QuantumModule, QuantumDevice, MeasureAll, PauliZ, GeneralEncoder, RandomLayer

class QuanvolutionEnhanced(nn.Module):
    """
    Hybrid Quanvolution network with a learnable quantum filter and optional attention weighting.
    """

    def __init__(self, n_wires: int = 4, n_ops: int = 8, use_attention: bool = False):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = RandomLayer(n_ops=n_ops, wires=list(range(self.n_wires)))
        self.measure = MeasureAll(PauliZ)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(4 * 14 * 14, 4 * 14 * 14),
                nn.Softmax(dim=-1)
            )
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = QuantumDevice(self.n_wires, bsz=bsz, device=device)
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
        features = torch.cat(patches, dim=1)
        if self.use_attention:
            weights = self.attention(features)
            features = features * weights
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionEnhanced"]
