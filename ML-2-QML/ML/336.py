"""Classical PyTorch classifier with a ResNet block and a simple sigmoid head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation used as a classical head."""

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
    """Linear layer followed by a sigmoid activation."""

    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = self.linear(inputs)
        return HybridFunction.apply(logits, self.shift)


class BasicBlock(nn.Module):
    """Simplified ResNet block with two 3×3 convolutions and a skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = (stride!= 1 or in_channels!= out_channels) and nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) or nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class QCNet(nn.Module):
    """CNN with a ResNet block followed by a classical sigmoid head."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.resblock = BasicBlock(6, 15, stride=2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Compute flattened feature size after conv/pool layers
        dummy_input = torch.zeros(1, 3, 32, 32)
        x = self._forward_features(dummy_input)
        flattened_size = x.shape[1]
        self.fc1 = nn.Linear(flattened_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(1, shift=0.0)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = self.resblock(x)
        x = self.pool(x)
        x = self.drop2(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._forward_features(inputs)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)


__all__ = ["HybridFunction", "Hybrid", "BasicBlock", "QCNet"]
