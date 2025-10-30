"""HybridQuantumNAT – classical implementation with optional quantum FCL."""
from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from typing import Iterable


def FCL() -> nn.Module:
    """
    Stand‑in for a fully‑connected quantum layer.
    Returns an nn.Module with a ``run`` method that mimics a quantum expectation
    value.  The method is intentionally non‑differentiable, serving as a
    placeholder for a real quantum circuit.
    """
    class FullyConnectedLayer(nn.Module):
        def __init__(self, n_features: int = 1) -> None:
            super().__init__()
            self.linear = nn.Linear(n_features, 1)

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
            expectation = torch.tanh(self.linear(values)).mean(dim=0)
            return expectation.detach().numpy()

    return FullyConnectedLayer()


class HybridQuantumNAT(nn.Module):
    """
    Classical CNN followed by either a fully‑connected linear head or a quantum
    fully‑connected layer (FCL).  The ``use_quantum`` flag selects the head type.
    """

    def __init__(self, use_quantum: bool = False, n_features: int = 4) -> None:
        super().__init__()
        self.use_quantum = use_quantum
        self.n_features = n_features

        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        if self.use_quantum:
            self.qcl = FCL()
        else:
            self.fc = nn.Sequential(
                nn.Linear(16 * 7 * 7, 64),
                nn.ReLU(),
                nn.Linear(64, self.n_features),
            )

        self.norm = nn.BatchNorm1d(self.n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flattened = features.view(bsz, -1)

        if self.use_quantum:
            # Run the quantum-inspired FCL on each sample
            out = torch.zeros(bsz, self.n_features, device=x.device)
            for i in range(bsz):
                thetas = flattened[i].tolist()
                expectation = self.qcl.run(thetas)  # returns numpy array
                out[i] = torch.tensor(expectation, device=x.device)
        else:
            out = self.fc(flattened)

        return self.norm(out)


__all__ = ["HybridQuantumNAT"]
