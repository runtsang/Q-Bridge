"""HybridClassifier: Classical PyTorch implementation with quantum-inspired feature map and activation.

This module extends the original hybrid classifier by adding a polynomial
feature expansion and a parametric sigmoid activation that mimics a
quantum expectation layer.  The architecture remains a convolutional
backbone followed by a dense head, but the final layer now uses a
custom autograd function that can be trained end‑to‑end with standard
optimizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QuantumSigmoid(torch.autograd.Function):
    """Differentiable parametric sigmoid that approximates a quantum expectation."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
        ctx.alpha = alpha
        ctx.beta = beta
        out = 0.5 * (1 + torch.tanh(alpha * inputs + beta))
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (out,) = ctx.saved_tensors
        grad = grad_output * ctx.alpha * (1 - out**2)
        return grad, None, None


class QuantumFeatureMap(nn.Module):
    """Polynomial feature expansion that mimics a quantum feature map."""

    def __init__(self, in_features: int, out_features: int, poly_order: int = 3) -> None:
        super().__init__()
        self.poly_order = poly_order
        self.linear = nn.Linear(in_features * poly_order, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _ = x.shape
        poly = [x]
        for p in range(2, self.poly_order + 1):
            poly.append(x ** p)
        poly_features = torch.cat(poly, dim=1)
        return self.linear(poly_features)


class HybridClassifier(nn.Module):
    """Convolutional network with a quantum-inspired head."""

    def __init__(self, alpha: float = 1.0, beta: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.feature_map = QuantumFeatureMap(in_features=1, out_features=84, poly_order=3)
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x).squeeze(-1)  # shape: (batch,)
        x = self.feature_map(x.unsqueeze(1)).squeeze(-1)  # shape: (batch, 84)
        x = x.mean(dim=1)  # shape: (batch,)
        probs = QuantumSigmoid.apply(x, self.alpha, self.beta)
        return torch.stack([probs, 1 - probs], dim=-1)


__all__ = ["QuantumSigmoid", "QuantumFeatureMap", "HybridClassifier"]
