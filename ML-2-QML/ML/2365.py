"""Combined classical-quantum binary classifier with kernel augmentation.

This module provides a hybrid CNN that optionally augments the quantum
expectation head with a classical RBF kernel.  The kernel is evaluated
on a reduced feature space and its output is fused with the quantum
probability to produce the final prediction.  The design follows the
original seed while adding a scalable kernel component that can be
switched on or off at initialization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Sequence

# Classical kernel components
class KernalAnsatz(nn.Module):
    """Classical RBF kernel ansatz."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """Wrapper around :class:`KernalAnsatz` for compatibility."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# Hybrid components
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
    """CNN-based binary classifier with optional classical kernel augmentation."""
    def __init__(self, use_kernel: bool = False, kernel_gamma: float = 1.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.kernel_input = nn.Linear(84, 4)  # reduce to 4‑dim for kernel
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(1, shift=0.0)

        self.use_kernel = use_kernel
        if use_kernel:
            # 10 support vectors in the reduced 4‑dim space
            self.support_vectors = nn.Parameter(torch.randn(10, 4), requires_grad=False)
            self.kernel = Kernel(kernel_gamma)
            self.kernel_classifier = nn.Linear(10, 1)

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
        # hybrid head
        hybrid_probs = self.hybrid(x)

        if self.use_kernel:
            # kernel head
            x_k = self.kernel_input(x)
            kernel_vals = torch.stack([self.kernel(x_k, sv) for sv in self.support_vectors])
            kernel_logits = self.kernel_classifier(kernel_vals.unsqueeze(0))
            kernel_probs = torch.sigmoid(kernel_logits)
            probs = 0.5 * hybrid_probs + 0.5 * kernel_probs
        else:
            probs = hybrid_probs

        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = ["KernalAnsatz", "Kernel", "kernel_matrix", "HybridFunction", "Hybrid", "QCNet"]
