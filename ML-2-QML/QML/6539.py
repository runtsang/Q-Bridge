"""Hybrid quantum‑classical quanvolution model with depthwise separable conv and a parameterised quantum patch encoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq

class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise conv followed by pointwise conv."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

class QuantumPatchEncoder(tq.QuantumModule):
    """Parameterised quantum encoder for 2x2 patches."""
    def __init__(self):
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder([
            {"input_idx": [0], "func": "ry", "wires": [0]},
            {"input_idx": [1], "func": "ry", "wires": [1]},
            {"input_idx": [2], "func": "ry", "wires": [2]},
            {"input_idx": [3], "func": "ry", "wires": [3]},
        ])
        self.entangle = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack([
                    x[:, r, c],
                    x[:, r, c+1],
                    x[:, r+1, c],
                    x[:, r+1, c+1],
                ], dim=1)
                self.encoder(qdev, data)
                self.entangle(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

class QuanvolutionEnhanced(nn.Module):
    """Hybrid quantum‑classical model with depthwise separable conv and quantum patch encoder."""
    def __init__(self, in_channels: int = 1, num_classes: int = 10):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.q_encoder = QuantumPatchEncoder()
        self.fusion = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.classifier = nn.Linear(128 * 7 * 7, num_classes)
        self.regressor = nn.Linear(128 * 7 * 7, 1)
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out_ds = self.ds_conv(x)
        out_q = self.q_encoder(x)
        out_q = out_q.view(x.size(0), 4, 14, 14)
        out = torch.cat([out_ds, out_q], dim=1)
        out = F.relu(self.fusion(out))
        out = F.adaptive_avg_pool2d(out, (7, 7))
        out_flat = out.view(out.size(0), -1)
        logits = self.classifier(out_flat)
        aux = self.regressor(out_flat)
        return logits, aux

__all__ = ["QuanvolutionEnhanced", "DepthwiseSeparableConv", "QuantumPatchEncoder"]
