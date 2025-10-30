"""Quanvolution dual‑branch filter with a true quantum kernel."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class QuantumConv2d(tq.QuantumModule):
    """Quantum convolutional layer applying a parameterised circuit to 2×2 patches."""
    def __init__(self, n_wires: int = 4, n_ops: int = 8, seed: int | None = None):
        super().__init__()
        self.n_wires = n_wires
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=n_ops, wires=list(range(self.n_wires)), seed=seed)
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
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class ClassicalBranch(nn.Module):
    """Classical depthwise separable conv + channel attention branch."""
    def __init__(self, in_channels: int = 1, out_channels: int = 4):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   stride=2, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.attn_pool = nn.AdaptiveAvgPool2d(1)
        self.attn_fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 16, out_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pointwise(self.depthwise(x))
        b, c, _, _ = x.shape
        y = self.attn_pool(x).view(b, c)
        y = self.attn_fc(y).view(b, c, 1, 1)
        return (x * y).view(b, -1)

class QuanvolutionDualFilter(tq.QuantumModule):
    """Dual‑branch filter: classical + true quantum branch."""
    def __init__(self, in_channels: int = 1, out_features: int = 4, n_wires: int = 4, n_ops: int = 8):
        super().__init__()
        self.classical = ClassicalBranch(in_channels, out_features)
        self.quantum = QuantumConv2d(n_wires=n_wires, n_ops=n_ops)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_feat = self.classical(x)
        q_feat = self.quantum(x)
        return torch.cat([cls_feat, q_feat], dim=1)

class QuanvolutionDualClassifier(nn.Module):
    """Classifier on top of the dual‑branch filter."""
    def __init__(self, num_classes: int = 10, in_channels: int = 1, out_features: int = 4):
        super().__init__()
        self.filter = QuanvolutionDualFilter(in_channels, out_features)
        self.linear = nn.Linear(2 * out_features * 14 * 14, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.filter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionDualFilter", "QuanvolutionDualClassifier"]
