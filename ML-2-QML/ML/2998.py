import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AdaptiveLinear(nn.Module):
    """Linear layer with a learnable gate to enable or disable the transform."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.gate = nn.Parameter(torch.ones(1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) * self.gate

class ClassicalHybridHead(nn.Module):
    """Classical approximation of a quantum expectation head using a sigmoid."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.linear(x)
        return torch.sigmoid(logits + self.shift)

class EstimatorQNN(nn.Module):
    """
    A flexible estimator that can operate purely classically, with a
    classical hybrid head, or with a quantum‑inspired hybrid head.
    The architecture is a fully connected feed‑forward network followed
    by either a linear layer (regression), a sigmoid head (classification),
    or a quantum‑inspired hybrid head.
    """
    def __init__(self,
                 in_features: int = 2,
                 hidden_sizes: tuple[int,...] = (8, 4),
                 activation: nn.Module = nn.Tanh(),
                 shift: float = 0.0,
                 quantum: bool = False,
                 classification: bool = False) -> None:
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(activation)
            prev = h
        self.net = nn.Sequential(*layers)
        self.quantum = quantum
        self.classification = classification
        if quantum:
            self.head = ClassicalHybridHead(prev, shift=shift)
        else:
            self.head = nn.Linear(prev, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        out = self.head(x)
        if self.classification:
            prob = torch.sigmoid(out)
            return torch.cat((prob, 1 - prob), dim=-1)
        return out

__all__ = ["EstimatorQNN", "AdaptiveLinear", "ClassicalHybridHead"]
