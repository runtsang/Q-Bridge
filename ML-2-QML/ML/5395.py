from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid head that mimics a quantum expectation."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Simple linear head with a sigmoid activation."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(logits, self.shift)

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch extractor inspired by the quanvolution example."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier that uses the quanvolution filter followed by a linear head."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

class QuantumClassifierModel(nn.Module):
    """Fully classical network mirroring the quantum architecture."""
    def __init__(self, in_channels: int = 3, num_classes: int = 2, shift: float = 0.0):
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # Fully‑connected head
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        # Hybrid sigmoid head
        self.hybrid = Hybrid(self.fc3.out_features, shift=shift)

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
        probabilities = self.hybrid(x)
        return torch.cat((probabilities, 1 - probabilities), dim=-1)

class RBFKernel(nn.Module):
    """Radial basis function kernel for classical similarity."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

def build_classifier_circuit(num_features: int, depth: int) -> tuple:
    """
    Construct a hybrid classical classifier mirroring the quantum interface.
    Parameters are retained for API compatibility but ignored in the classical implementation.
    Returns:
        network (nn.Module): The hybrid model.
        encoding (list): Empty for classical.
        weight_sizes (list): Number of parameters per layer.
        observables (list): Output class indices.
    """
    network = QuantumClassifierModel()
    encoding: list = []
    weight_sizes = [p.numel() for p in network.parameters()]
    observables = [0, 1]
    return network, encoding, weight_sizes, observables

__all__ = [
    "HybridFunction",
    "Hybrid",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "QuantumClassifierModel",
    "RBFKernel",
    "build_classifier_circuit",
]
