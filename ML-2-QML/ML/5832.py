"""UnifiedHybridEstimator: classical-only hybrid model for regression and classification.

The design merges the minimal feed‑forward regressor (EstimatorQNN) and the deep CNN‑based binary classifier (QCNet) into a single, fully‑classical architecture.  It introduces a hybrid‑like activation that replaces the quantum expectation layer, enabling a single model to perform regression or binary classification.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HybridFunction", "Hybrid", "UnifiedHybridEstimator"]

class HybridFunction(torch.autograd.Function):
    """Differentiable hybrid‑like activation that mimics a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.shift = shift
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Simple hybrid head that replaces the quantum circuit."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return HybridFunction.apply(logits, self.shift)

class UnifiedHybridEstimator(nn.Module):
    """CNN backbone followed by a hybrid head.  Supports regression or binary classification."""
    def __init__(self, output_dim: int = 1, shift: float = 0.0):
        super().__init__()
        self.output_dim = output_dim
        self.shift = shift

        # CNN backbone identical to QCNet
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Assume input images of size 224x224; flatten size 55815
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        self.hybrid = Hybrid(self.fc3.out_features, shift=self.shift)

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
        x = self.fc3(x)
        logits = self.hybrid(x)

        if self.output_dim == 1:
            # Regression
            return logits
        else:
            # Binary classification
            probs = torch.sigmoid(logits)
            return torch.cat([probs, 1 - probs], dim=-1)
