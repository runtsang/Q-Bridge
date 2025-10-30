from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """
    Differentiable sigmoid head that mimics a quantum expectation layer.
    Uses a simple parameter shift for the bias term.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        logits = inputs + shift
        outputs = torch.sigmoid(logits)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None


class Hybrid(nn.Module):
    """
    Lightweight hybrid layer: a linear projection followed by a
    differentiable sigmoid.  The linear layer learns a bias that
    can be seen as a classical surrogate for a quantum expectation.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=True)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = self.linear(inputs)
        return HybridFunction.apply(logits, self.shift)


class QCNet(nn.Module):
    """
    Classical CNN with a hybrid sigmoid head, suitable for binary
    classification.  The architecture includes:
        • 3 convolutional layers with batch‑norm and ReLU
        • Max‑pooling and dropout for regularisation
        • Adaptive average pooling to guarantee a fixed feature size
        • 3 fully‑connected layers ending with the Hybrid head
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pool  = nn.MaxPool2d(2)
        self.drop  = nn.Dropout2d(p=0.5)
        self.adapt = nn.AdaptiveAvgPool2d((4, 4))

        # Feature size after adaptive pooling: 128 * 4 * 4 = 2048
        self.fc1   = nn.Linear(128 * 4 * 4, 512)
        self.fc2   = nn.Linear(512, 128)
        self.fc3   = nn.Linear(128, 1)

        self.hybrid = Hybrid(1, shift=0.0)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(inputs)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop(x)
        x = self.adapt(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probs = self.hybrid(x)
        return torch.cat((probs, 1 - probs), dim=-1)


__all__ = ["HybridFunction", "Hybrid", "QCNet"]
