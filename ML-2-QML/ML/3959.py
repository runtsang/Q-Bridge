"""Hybrid classical-quantum binary classification: classical variant.

This module defines a PyTorch-based classifier that mirrors the structure
of the quantum counterpart. It uses a modular factory to build a
feed‑forward network and a simple sigmoid head, enabling easy scaling
through the `depth` argument.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, List

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, List[int], List[int], List[int]]:
    """
    Construct a feed‑forward classifier and metadata similar to the quantum variant.
    Returns the network, encoding indices, weight sizes, and observables (class indices).
    """
    layers: List[nn.Module] = []
    in_dim = num_features
    encoding: List[int] = list(range(num_features))
    weight_sizes: List[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables: List[int] = list(range(2))
    return network, encoding, weight_sizes, observables

class HybridFunction(torch.autograd.Function):
    """
    Classical differentiable head that emulates the quantum expectation.
    Currently implements a sigmoid activation with an optional shift.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        ctx.shift = shift
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        outputs, = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1.0 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """
    Hybrid layer that forwards activations through a classical head.
    """
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(logits, self.shift)

class QCNet(nn.Module):
    """
    Classical convolutional network followed by a parameterised classical head.
    """
    def __init__(self, depth: int = 1, shift: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        self.hybrid = Hybrid(self.fc3.out_features, shift)

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
        p = self.hybrid(x).squeeze(-1)
        probs = torch.stack([p, 1 - p], dim=-1)
        return probs

__all__ = ["build_classifier_circuit", "HybridFunction", "Hybrid", "QCNet"]
