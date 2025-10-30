from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SamplerModule(nn.Module):
    """A lightweight classical sampler that maps 2‑dimensional inputs to a probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)


class QuantumFullyConnectedLayer(nn.Module):
    """Mimics a quantum fully‑connected layer with a simple expectation‑value proxy."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


class HybridSamplerQNN(nn.Module):
    """Combines the classical sampler with a quantum‑style fully‑connected layer."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.sampler = SamplerModule()
        self.fcl = QuantumFullyConnectedLayer(n_features)

    def forward(self, inputs: torch.Tensor, thetas: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
        probs = self.sampler(inputs)
        expectation = self.fcl.run(thetas)
        return probs.detach().numpy(), expectation


__all__ = ["HybridSamplerQNN"]
