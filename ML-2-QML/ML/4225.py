"""Hybrid fraud detection model combining classical dense layers and a photonic quantum head."""

from __future__ import annotations

import torch
from torch import nn
from typing import Sequence

import quantum_fraud

def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate data for regression or classification tasks.
    Mirrors the generation logic from QuantumRegression.py.
    """
    x = torch.rand(samples, num_features) * 2 - 1  # uniform in [-1, 1]
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y

class SuperpositionDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper around the superposition data generator.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return {"states": self.features[idx], "target": self.labels[idx]}

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud detection model.
    - Classical dense network learns feature embeddings.
    - Quantum photonic head produces a differentiable expectation value.
    - Final probability is obtained via a sigmoid on the sum of classical logits and quantum expectation.
    """

    def __init__(self, input_dim: int = 2, hidden_dims: Sequence[int] = (64, 32)) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # raw logits before quantum head
        self.classical = nn.Sequential(*layers)

        # Quantum head: uses a photonic circuit with 2 modes
        self.quantum_head = quantum_fraud.QPhotonicHead()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical embedding
        logits = self.classical(x)
        # Quantum expectation
        # Detach to avoid double gradient flow
        q_expect = self.quantum_head(logits.detach())
        # Combine: sigmoid of sum
        probs = torch.sigmoid(logits + q_expect)
        return torch.cat([probs, 1 - probs], dim=-1)
