"""
Hybrid Quantum Binary Classifier – Classical (PyTorch) counterpart.

This module implements a CNN backbone followed by a lightweight
Hybrid head that emulates the quantum expectation layer with a
differentiable sigmoid.  The head can be swapped out with an
actual quantum circuit at runtime, enabling easy ablation studies.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridSigmoidFunction(torch.autograd.Function):
    """Differentiable sigmoid activation mimicking a quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        return grad_output * outputs * (1 - outputs), None


class HybridHead(nn.Module):
    """Simple linear head with optional bias shift."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=True)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x).squeeze(-1)
        return HybridSigmoidFunction.apply(logits, self.shift)


class HybridQuantumBinaryClassifier(nn.Module):
    """
    CNN backbone + hybrid head.  The architecture mirrors the quantum
    version but is fully classical, making it suitable for quick
    prototyping and ablation.
    """
    def __init__(
        self,
        in_channels: int = 3,
        conv_features: list[int] | None = None,
        fc_features: list[int] | None = None,
        shift: float = 0.0,
        dropout: float = 0.5,
    ):
        super().__init__()
        if conv_features is None:
            conv_features = [6, 15]
        if fc_features is None:
            fc_features = [120, 84]

        # Convolutional backbone
        layers = []
        in_ch = in_channels
        for out_ch in conv_features:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=5 if out_ch==6 else 3,
                                    stride=2, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
            layers.append(nn.Dropout2d(p=dropout))
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)

        # Fully connected part
        flat_dim = 55815  # fixed for 32x32 RGB input after conv layers
        fc_layers = []
        in_fc = flat_dim
        for out_fc in fc_features:
            fc_layers.append(nn.Linear(in_fc, out_fc))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(p=dropout))
            in_fc = out_fc
        fc_layers.append(nn.Linear(in_fc, 1))
        self.fcs = nn.Sequential(*fc_layers)

        # Hybrid head
        self.hybrid = HybridHead(1, shift=shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fcs(x)
        probs = self.hybrid(x)
        return torch.stack((probs, 1 - probs), dim=-1)


__all__ = ["HybridSigmoidFunction", "HybridHead", "HybridQuantumBinaryClassifier"]
