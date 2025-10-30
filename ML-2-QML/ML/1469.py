"""Classical regression model with feature engineering and a flexible dataset."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class HybridRegressor(nn.Module):
    """A multi‑layer perceptron with optional polynomial feature expansion and dropout."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int,...] = (64, 32),
        dropout: float = 0.0,
        use_poly: bool = False,
        poly_degree: int = 2,
    ):
        super().__init__()
        self.use_poly = use_poly
        self.poly_degree = poly_degree
        in_dim = input_dim
        if use_poly:
            in_dim = input_dim * poly_degree
        layers = []
        prev_dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def _poly_expand(self, x: torch.Tensor) -> torch.Tensor:
        """Return tensor of all monomials up to self.poly_degree for each feature individually."""
        features = [x]
        for d in range(2, self.poly_degree + 1):
            features.append(torch.pow(x, d))
        return torch.cat(features, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_poly:
            x = self._poly_expand(x)
        return self.net(x).squeeze(-1)

class HybridRegressionDataset(Dataset):
    """Generate synthetic data from a mix of sinusoid and polynomial signals."""
    def __init__(
        self,
        samples: int,
        input_dim: int,
        seed: int | None = None,
        noise_std: float = 0.1,
    ):
        rng = np.random.default_rng(seed)
        self.features = rng.uniform(-1.0, 1.0, size=(samples, input_dim)).astype(np.float32)
        coeff_lin = rng.uniform(-1, 1, size=(input_dim, 1)).ravel()
        coeff_quad = rng.normal(size=(input_dim, 1)).ravel()
        self.labels = (
            np.sin(self.features @ coeff_lin) + 0.5 * (self.features @ coeff_quad) ** 2
        ).astype(np.float32)
        self.labels += rng.normal(scale=noise_std, size=samples).astype(np.float32)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["HybridRegressor", "HybridRegressionDataset"]
