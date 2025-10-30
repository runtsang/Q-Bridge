import numpy as np
import torch
import torch.nn as nn
from typing import Sequence, Iterable, Tuple

class ClassicalRBFKernel(nn.Module):
    """Differentiable RBF kernel implemented with PyTorch."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(gamma, dtype=torch.float32))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        sqdist = torch.sum(diff * diff, dim=-1)
        return torch.exp(-self.gamma * sqdist)

class FeedForwardClassifier(nn.Module):
    """Feed‑forward network mirroring the classical classifier factory."""
    def __init__(self, num_features: int, depth: int):
        super().__init__()
        layers = []
        in_dim = num_features
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU())
            in_dim = num_features
        layers.append(nn.Linear(num_features, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class QuantumKernelClassifier(nn.Module):
    """Hybrid class providing both a classical RBF kernel and a feed‑forward classifier."""
    def __init__(self, num_features: int, depth: int = 3, gamma: float = 1.0):
        super().__init__()
        self.kernel = ClassicalRBFKernel(gamma)
        self.classifier = FeedForwardClassifier(num_features, depth)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        a_batch = torch.stack(a)
        b_batch = torch.stack(b)
        with torch.no_grad():
            mat = self.kernel(a_batch, b_batch).cpu().numpy()
        return mat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return a feed‑forward network and metadata mimicking the quantum build function."""
    net = FeedForwardClassifier(num_features, depth)
    encoding = list(range(num_features))
    weight_sizes = []
    for layer in net.network:
        if isinstance(layer, nn.Linear):
            weight_sizes.append(layer.weight.numel() + layer.bias.numel())
    observables = list(range(2))
    return net, encoding, weight_sizes, observables

__all__ = ["ClassicalRBFKernel", "FeedForwardClassifier", "QuantumKernelClassifier", "build_classifier_circuit"]
