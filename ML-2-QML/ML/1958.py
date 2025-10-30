"""
Hybrid Quantum Binary Classifier – Classical Backend.
Provides a CNN feature extractor followed by a differentiable hybrid layer that
mimics a quantum expectation value. The implementation is modular and can
optionally add a classical auxiliary head for ensembling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridFunction(torch.autograd.Function):
    """
    Differentiable sigmoid head that simulates the quantum expectation value.
    Allows a learnable shift; gradient is computed analytically.
    """
    @staticmethod
    def forward(ctx, logits: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.shift = shift
        ctx.save_for_backward(logits)
        probs = torch.sigmoid(logits + shift)
        return probs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        logits, = ctx.saved_tensors
        shift = ctx.shift
        probs = torch.sigmoid(logits + shift)
        grad_logits = grad_output * probs * (1.0 - probs)
        return grad_logits, None

class Hybrid(nn.Module):
    """
    Hybrid head that applies a linear layer followed by HybridFunction.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        probs = HybridFunction.apply(logits, self.shift)
        return probs

class AuxiliaryClassifier(nn.Module):
    """
    Optional lightweight classifier that operates on the same feature vector
    for ensembling or calibration.
    """
    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))

class HybridQCNet(nn.Module):
    """
    CNN feature extractor followed by a quantum‑mimicking hybrid head.
    An auxiliary classical head can be enabled for fusion.
    """
    def __init__(self, aux: bool = False) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1),  # 3×224×224 → 6×110×110
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                # 6×110×110 → 6×55×55
            nn.Dropout2d(p=0.2),

            nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1),# 15×27×27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                # 15×13×13
            nn.Dropout2d(p=0.5),
        )
        # Flattened size: 15 * 13 * 13 = 2535
        self.flatten_dim = 15 * 13 * 13

        # Fully connected backbone
        self.classifier = nn.Sequential(
            nn.Linear(self.flatten_dim, 120),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        )
        # Hybrid head
        self.hybrid = Hybrid(84, shift=0.0)

        # Optional auxiliary head
        self.aux = AuxiliaryClassifier(84) if aux else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        primary = self.hybrid(x)
        if self.aux is not None:
            aux_out = self.aux(x)
            # Simple soft voting fusion
            out = (primary + aux_out) / 2.0
        else:
            out = primary
        # Convert to 2‑class probabilities
        return torch.cat([out, 1 - out], dim=-1)

__all__ = ["HybridFunction", "Hybrid", "AuxiliaryClassifier", "HybridQCNet"]
