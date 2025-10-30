"""Enhanced classical regression with residual attention and multi‑output support."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


def generate_superposition_data(
    num_features: int,
    samples: int,
    output_dim: int = 1,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic dataset based on a sinusoidal superposition.

    The input features are sampled uniformly from ``[-1, 1]``.  For each example
    we compute a base angle ``theta`` as the sum of all features and produce
    ``output_dim`` targets that are sinusoidal functions of ``theta`` and a
    random phase.  The return shape is ``(samples, num_features)`` for the
    features and ``(samples, output_dim)`` for the targets.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    thetas = x.sum(axis=1)
    outputs = np.zeros((samples, output_dim), dtype=np.float32)
    for i in range(output_dim):
        phi = rng.uniform(0, 2 * np.pi, size=samples)
        outputs[:, i] = np.sin(2 * thetas) * np.cos(phi)
    return x, outputs.astype(np.float32)


class ResidualBlock(nn.Module):
    """A residual block with a channel‑wise attention gate."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_dim, out_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(out_dim // 4, out_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.mlp(x)
        att = self.attention(out.unsqueeze(-1)).squeeze(-1)
        out = out * att
        return out + residual


class HybridRegressionModel(nn.Module):
    """Classical regression model with residual blocks and attention.

    The network consists of an initial linear layer, followed by a stack of
    residual blocks.  Each block applies a small MLP, a channel‑wise
    attention module, and adds the input back to the output.  The final head
    maps to ``output_dim`` regression targets.
    """

    def __init__(
        self,
        num_features: int,
        output_dim: int = 1,
        *,
        hidden_dim: int = 32,
        n_blocks: int = 3,
    ):
        super().__init__()
        self.input_layer = nn.Linear(num_features, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, hidden_dim) for _ in range(n_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(state_batch)
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x)


class RegressionDataset(Dataset):
    """Dataset that returns feature vectors and multi‑output targets.

    The ``target`` field is a 1‑D tensor of length ``output_dim``.
    """

    def __init__(
        self,
        samples: int,
        num_features: int,
        output_dim: int = 1,
        *,
        seed: int | None = None,
    ):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, output_dim=output_dim, seed=seed
        )

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
