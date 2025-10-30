"""Classical PyTorch network with a residual dense head for binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """A residual block that applies dropout and a ReLU activation.

    If ``in_features`` and ``out_features`` differ, a linear shortcut
    aligns the dimensions before adding the residual.
    """
    def __init__(self, in_features: int, out_features: int,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=True)
        if in_features!= out_features:
            self.shortcut = nn.Linear(in_features, out_features)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.linear(x)
        out = self.dropout(out)
        out = self.relu(out)
        return self.relu(out + residual)


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that emulates a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.save_for_backward(inputs)
        ctx.shift = shift
        return torch.sigmoid(inputs + shift)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        inputs, = ctx.saved_tensors
        sigmoid = torch.sigmoid(inputs + ctx.shift)
        grad_inputs = grad_output * sigmoid * (1 - sigmoid)
        return grad_inputs, None


class Hybrid(nn.Module):
    """Simple dense head that replaces the quantum circuit in the original model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)


class QCNet(nn.Module):
    """CNN-based binary classifier with a residual dense head."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Dense head with residual block
        self.fc1 = nn.Linear(55815, 120)
        self.residual = ResidualDenseBlock(120, 120, dropout=0.3)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Classical hybrid head
        self.hybrid = Hybrid(self.fc3.out_features, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.residual(x)
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["ResidualDenseBlock", "HybridFunction", "Hybrid", "QCNet"]
