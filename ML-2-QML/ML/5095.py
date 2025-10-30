"""Hybrid classical kernel, estimator, sampler and classifier."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "HybridKernelModel",
    "RBFKernel",
    "EstimatorNN",
    "SamplerNN",
    "ClassifierBuilder",
]


class RBFKernel(nn.Module):
    """Exponentiated squared distance kernel.  Parameters are learned
    or fixed externally, allowing a direct comparison with a quantum
    kernel that uses the same functional form."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


class EstimatorNN(nn.Module):
    """Simple feed‑forward regressor that mirrors the EstimatorQNN example."""
    def __init__(self, input_dim: int = 2, hidden: list[int] = (8, 4)) -> None:
        super().__init__()
        layers = []
        in_dim = input_dim
        for h in hidden:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


class SamplerNN(nn.Module):
    """Classification sampler that outputs a probability distribution."""
    def __init__(self, input_dim: int = 2, hidden: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


class ClassifierBuilder:
    """Builds a shallow feed‑forward classifier and returns metadata
    that mimics the quantum interface (encoding indices, weight sizes,
    observables)."""
    def __init__(self, num_features: int, depth: int) -> None:
        self.num_features = num_features
        self.depth = depth

    def build(self) -> tuple[nn.Module, list[int], list[int], list[int]]:
        layers = []
        in_dim = self.num_features
        weight_sizes = []
        for _ in range(self.depth):
            linear = nn.Linear(in_dim, self.num_features)
            layers.extend([linear, nn.ReLU()])
            weight_sizes.append(linear.weight.numel() + linear.bias.numel())
            in_dim = self.num_features
        head = nn.Linear(in_dim, 2)
        layers.append(head)
        weight_sizes.append(head.weight.numel() + head.bias.numel())
        network = nn.Sequential(*layers)

        encoding = list(range(self.num_features))
        observables = list(range(2))
        return network, encoding, weight_sizes, observables


class HybridKernelModel:
    """Container that exposes both classical and quantum utilities
    in a single API, facilitating side‑by‑side benchmarking."""
    def __init__(self, gamma: float = 1.0, depth: int = 2) -> None:
        self.kernel = RBFKernel(gamma)
        self.estimator = EstimatorNN()
        self.sampler = SamplerNN()
        self.classifier, self.encoding, self.weight_sizes, self.observables = (
            ClassifierBuilder(2, depth).build()
        )

    # ------------------------------------------------------------------
    # Classical utilities
    # ------------------------------------------------------------------
    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        return np.array([[self.kernel(x, y).item() for y in b] for x in a])

    def train_estimator(
        self,
        X: Sequence[torch.Tensor],
        y: Sequence[float],
        lr: float = 1e-3,
        epochs: int = 100,
    ) -> nn.Module:
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        optimizer = torch.optim.Adam(self.estimator.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.estimator(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        return self.estimator

    def sample(self, X: Sequence[torch.Tensor]) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32)
        probs = self.sampler(X)
        return probs.detach().numpy()

    def classify(self, X: Sequence[torch.Tensor]) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32)
        logits = self.classifier(X)
        return torch.argmax(logits, dim=-1).detach().numpy()
