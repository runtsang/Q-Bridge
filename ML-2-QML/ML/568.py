"""Hybrid regression module – classical side enriched with attention and optional ensemble support."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Tuple


def generate_superposition_data(num_features: int, samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce a dataset that mimics a superposition‑like relationship.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset that stores a pre‑generated superposition dataset."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


class AttentionBlock(nn.Module):
    """
    Lightweight self‑attention module that operates on the feature vector.
    """

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.scale = 1.0 / np.sqrt(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)
        attn = torch.nn.functional.softmax(
            torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1
        )
        out = torch.matmul(attn, v)
        return self.proj(out)


class QModel(nn.Module):
    """
    Classical regression network with optional ensemble and attention.
    """

    def __init__(self, num_features: int, n_estimators: int = 1):
        super().__init__()
        self.num_features = num_features
        self.n_estimators = n_estimators

        def _make_net() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(num_features, 32),
                nn.ReLU(),
                AttentionBlock(32),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        if self.n_estimators == 1:
            self.net = _make_net()
        else:
            self.estimators = nn.ModuleList([_make_net() for _ in range(self.n_estimators)])

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        if self.n_estimators == 1:
            return self.net(state_batch).squeeze(-1)
        else:
            preds = torch.stack([est(state_batch) for est in self.estimators], dim=0)
            return preds.mean(dim=0).squeeze(-1)


__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
