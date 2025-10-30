from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class SimpleAttention(nn.Module):
    'Self‑attention over patch features.'
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C]
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        q = q.reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        k = k.reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        v = v.reshape(B, N, self.heads, C // self.heads).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.out(out)

class AdvancedQuanvolutionFilter(tq.QuantumModule):
    'Quantum filter with a learnable parameterized circuit.'
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {'input_idx': [0], 'func': 'ry', 'wires': [0]},
                {'input_idx': [1], 'func': 'ry', 'wires': [1]},
                {'input_idx': [2], 'func': 'ry', 'wires': [2]},
                {'input_idx': [3], 'func': 'ry', 'wires': [3]},
            ]
        )
        # Trainable rotation angles
        self.theta = nn.Parameter(torch.randn(4))
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
                # Apply trainable rotations
                qdev.ry(self.theta[0], 0)
                qdev.ry(self.theta[1], 1)
                qdev.ry(self.theta[2], 2)
                qdev.ry(self.theta[3], 3)
                # Entanglement
                qdev.cx(0, 1)
                qdev.cx(2, 3)
                qdev.cx(1, 2)
                measurement = self.measure(qdev)
                patches.append(measurement)
        # Stack patches: [bsz, 196, 4]
        features = torch.stack(patches, dim=1)
        return features

class AdvancedQuanvolutionClassifier(nn.Module):
    'Classifier using the advanced quanvolution quantum filter and attention.'
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.qfilter = AdvancedQuanvolutionFilter()
        self.attention = SimpleAttention(dim=4, heads=4)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # [B, 196, 4]
        features = self.attention(features)  # [B, 196, 4]
        features = features.reshape(x.size(0), -1)  # [B, 4*196]
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ['AdvancedQuanvolutionFilter', 'AdvancedQuanvolutionClassifier', 'SimpleAttention']
