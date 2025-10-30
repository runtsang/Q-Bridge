"""Classical CNN classifier for binary classification.

This module implements a lightweight yet expressive CNN with a
parameterised sigmoid head.  It is deliberately simple so that it can
serve as a drop‑in replacement for the hybrid version during
ablation studies.  The architecture mirrors the original
`QCNet` but with additional batch normalisation and a
well‑tuned dropout schedule.  All tensors are kept on the same
device as the model for seamless GPU acceleration.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassicalHybridFunction(torch.autograd.Function):
    """Direct sigmoid mapping that mimics the quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.shift = shift
        output = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (output,) = ctx.saved_tensors
        grad_input = grad_output * output * (1 - output)
        return grad_input, None

class ClassicalHybrid(nn.Module):
    """Dense head that replaces the quantum layer."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return ClassicalHybridFunction.apply(logits, self.shift)

class HybridBinaryClassifier(nn.Module):
    """CNN + sigmoid head for binary classification."""
    def __init__(self, device: torch.device | str | None = None) -> None:
        super().__init__()
        self.device = torch.device(device or "cpu")
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop1 = nn.Dropout2d(p=0.25)
        self.drop2 = nn.Dropout2d(p=0.5)

        dummy = torch.zeros(1, 3, 32, 32, device=self.device)
        tmp = self._forward_conv(dummy)
        flat_features = tmp.shape[1]

        self.fc1 = nn.Linear(flat_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.hybrid = ClassicalHybrid(1, shift=0.0)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.drop2(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_conv(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)

    def evaluate(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience evaluation on a batch of inputs."""
        self.eval()
        with torch.no_grad():
            return self(inputs)

__all__ = ["HybridBinaryClassifier"]
