"""Classical hybrid kernel and classifier.

This module implements a hybrid kernel classifier that uses a classical
RBF kernel and a dense head.  The architecture is compatible with the
original `QuantumKernelMethod` seed, but can be swapped with the
quantum counterpart defined in the QML module.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

# ---------- Classical kernel utilities ----------
class KernalAnsatz(nn.Module):
    """Analytic RBF kernel implemented as a torch module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wraps the analytic RBF kernel and exposes a kernel_matrix."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    """Compute Gram matrix between two sets of tensors."""
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ---------- Hybrid classifier utilities ----------
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics the quantum expectation head."""
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

class Hybrid(nn.Module):
    """Simple dense head that replaces the quantum circuit in the original model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class QCNet(nn.Module):
    """CNN-based binary classifier mirroring the structure of the quantum model."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
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
        x = self.fc3(x)
        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)

    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return the 1‑D feature vector before the hybrid head."""
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
        return self.fc3(x)  # 1‑D feature vector

# ---------- Hybrid kernel classifier ----------
class HybridKernelClassifier(nn.Module):
    """
    A hybrid kernel classifier that combines a feature extractor,
    a kernel (classical RBF or quantum variational) and a head.
    """
    def __init__(self,
                 kernel_type: str = "rbf",
                 head_type: str = "dense",
                 gamma: float = 1.0,
                 shift: float = 0.0) -> None:
        super().__init__()
        # Feature extractor shared between classical and quantum heads
        self.extractor = QCNet()
        # Kernel selection
        if kernel_type == "rbf":
            self.kernel = Kernel(gamma)
        else:
            raise ValueError(f"Unsupported kernel_type {kernel_type}")
        # Head selection
        if head_type == "dense":
            self.head = Hybrid(1, shift=shift)
        else:
            raise ValueError(f"Unsupported head_type {head_type}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Extract 1‑D features from the CNN backbone
        features = self.extractor.extract_features(inputs)
        # Compute kernel similarity between features and themselves
        kernel_vals = self.kernel(features, features)
        # Pass through the head to obtain logits
        logits = self.head(kernel_vals)
        return torch.cat((logits, 1 - logits), dim=-1)

__all__ = [
    "KernalAnsatz",
    "Kernel",
    "kernel_matrix",
    "HybridFunction",
    "Hybrid",
    "QCNet",
    "HybridKernelClassifier",
]
