import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ClassicalScaling(nn.Module):
    """Learnable linear scaling that maps the FC output to a convenient range for the quantum head."""
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class HybridFunction(torch.autograd.Function):
    """Placeholder differentiable bridge. In the classical version, it simply returns the input."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        return inputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output

class Hybrid(nn.Module):
    """Base hybrid layer that forwards activations through the placeholder HybridFunction."""
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return HybridFunction.apply(inputs)

class QCNet(nn.Module):
    """CNN-based binary classifier mirroring the structure of the quantum model, with a classical scaling head."""
    def __init__(self, scaling_in_features: int = 84):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.scaling = ClassicalScaling(84, 1)
        self.hybrid = Hybrid()

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
        x = self.scaling(x)
        x = self.hybrid(x)
        return torch.cat((x, 1 - x), dim=-1)

__all__ = ["ClassicalScaling", "HybridFunction", "Hybrid", "QCNet"]
