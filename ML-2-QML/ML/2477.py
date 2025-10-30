"""Hybrid classical estimator that can incorporate quantum feature vectors."""
from __future__ import annotations

import torch
from torch import nn

class HybridEstimatorQNN(nn.Module):
    """
    A simple feed‑forward network that accepts 2‑dimensional classical inputs
    and an optional 1‑dimensional quantum feature vector.
    """
    def __init__(self, input_dim: int = 2, quantum_dim: int = 0,
                 hidden_dims: tuple[int,...] = (8, 4)) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.quantum_dim = quantum_dim
        total_dim = input_dim + quantum_dim

        layers = []
        prev = total_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.Tanh())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor,
                q_feat: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Classical feature tensor of shape (batch, input_dim).
        q_feat : torch.Tensor | None, optional
            Quantum feature tensor of shape (batch, quantum_dim).
        """
        if self.quantum_dim > 0 and q_feat is not None:
            x = torch.cat([x, q_feat], dim=-1)
        return self.net(x).squeeze(-1)

def EstimatorQNN() -> HybridEstimatorQNN:
    """
    Factory that returns a default hybrid estimator with 2 classical inputs
    and a 1‑dimensional quantum feature vector.
    """
    return HybridEstimatorQNN(input_dim=2, quantum_dim=1)

__all__ = ["HybridEstimatorQNN", "EstimatorQNN"]
