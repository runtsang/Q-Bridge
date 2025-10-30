"""Hybrid convolutional network with classical kernel and NAT approximations.

This module defines HybridConvNet, a drop‑in replacement for the
original hybrid quantum network.  It combines a classical convolutional
backbone, a classical RBF kernel (KernalAnsatz), a classical QFCModel
(NAT‑style feature extractor), and a hybrid sigmoid head that mimics
the quantum expectation layer.  The architecture is deliberately
compatible with the original Conv.py interface while adding richer
feature extraction and classification capabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Classical RBF kernel approximation
class KernalAnsatz(nn.Module):
    """Classical RBF kernel approximation."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around KernalAnsatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

# Classical Quantum‑NAT CNN approximation
class QFCModel(nn.Module):
    """Classical approximation of the Quantum‑NAT CNN."""
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)
        out = self.fc(flattened)
        return self.norm(out)

# Hybrid sigmoid head
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head mimicking quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class HybridConvNet(nn.Module):
    """Hybrid convolutional network with kernel and NAT feature extraction."""
    def __init__(self, shift: float = 0.0, kernel_gamma: float = 1.0) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Fully connected head
        self.fc1 = nn.Linear(55819, 120)  # 55815 conv features + 4 NAT features
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Classical kernel and NAT modules
        self.kernel = Kernel(gamma=kernel_gamma)
        self.nat = QFCModel()

        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Convolutional feature extraction
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)

        x = torch.flatten(x, 1)
        nat_features = self.nat(inputs[:, :1, :, :])  # use first channel for NAT
        combined = torch.cat([x, nat_features], dim=1)

        x = F.relu(self.fc1(combined))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        probs = HybridFunction.apply(x, self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["HybridConvNet"]
