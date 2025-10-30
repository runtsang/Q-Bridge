"""Combined classical-quantum hybrid model integrating quanvolution, QCNN, and quantum NAT ideas.

The model comprises a classical CNN, a quantum-inspired filter, a QCNN-style fully connected block,
and a hybrid sigmoid head.  It is designed to be drop‑in compatible with the original
``Quanvolution.py`` interface while providing a richer feature extraction pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumFilterApprox(nn.Module):
    """Classical approximation of a quantum filter using a small patchwise convolution
    followed by a random linear projection and batch‑normalisation.
    """

    def __init__(self, in_channels: int = 1, out_features: int = 4 * 14 * 14) -> None:
        super().__init__()
        # Encode 2×2 patches into 4‑channel feature maps
        self.patch_conv = nn.Conv2d(in_channels, 4, kernel_size=2, stride=2, bias=False)
        # Random projection mimicking a quantum measurement
        self.random_proj = nn.Linear(4 * 14 * 14, out_features, bias=False)
        self.bn = nn.BatchNorm1d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_conv(x)  # (bsz, 4, 14, 14)
        flattened = patches.view(patches.size(0), -1)
        projected = self.random_proj(flattened)
        return self.bn(projected)


class QCNNBlock(nn.Module):
    """Classical QCNN‑style fully‑connected block inspired by the quantum CNN helper."""

    def __init__(self, in_features: int = 4 * 14 * 14, hidden: int = 32) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(in_features, hidden), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.head = nn.Linear(hidden, 4)  # output 4 features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head for the classical model."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class QuanvolutionHybrid(nn.Module):
    """Hybrid model combining classical convolution, a quantum‑inspired filter,
    a QCNN‑style block, and a hybrid sigmoid head.
    """

    def __init__(self, in_channels: int = 1, num_classes: int = 10, shift: float = 0.0) -> None:
        super().__init__()
        # Classical CNN feature extractor
        self.classical = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Quantum‑inspired filter
        self.quantum = QuantumFilterApprox(in_channels=in_channels)
        # QCNN‑style transformation
        self.qcnn = QCNNBlock()
        # Final classifier
        self.classifier = nn.Linear(4, num_classes)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical feature path
        c_feat = self.classical(x).view(x.size(0), -1)
        # Quantum‑inspired filter path
        q_feat = self.quantum(x)
        # Concatenate
        combined = torch.cat([c_feat, q_feat], dim=1)
        # QCNN block
        qcnn_out = self.qcnn(combined)
        logits = self.classifier(qcnn_out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["QuanvolutionHybrid"]
