"""Hybrid classical regression module with optional quantum feature map.

The model accepts real or complex feature tensors.  When a complex tensor
is supplied, its real and imaginary parts are concatenated before being
passed to the linear layers.  The data generator can optionally apply a
user‑supplied feature map (e.g. a quantum circuit simulator) to the raw
inputs, enabling end‑to‑end training of quantum‑enhanced features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]

def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    feature_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce synthetic regression data.  If *feature_map* is supplied it is
    applied to the raw input; otherwise a simple sinusoidal target is
    generated.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vectors.
    samples : int
        Number of samples to generate.
    feature_map : Callable[[np.ndarray], np.ndarray] | None
        Function that transforms a (samples, num_features) array into
        another array of the same shape.  It can be a quantum circuit
        simulator that returns real‑valued expectations.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    if feature_map is not None:
        x = feature_map(x)
    y = np.sin(np.sum(x, axis=1)) + 0.05 * np.cos(2 * np.sum(x, axis=1))
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset of synthetic regression samples.  Returns a dictionary with
    keys ``states`` and ``target``.  ``states`` can be either real or
    complex tensors, depending on the data generator.
    """
    def __init__(
        self,
        samples: int,
        num_features: int,
        *,
        feature_map: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.features, self.labels = generate_superposition_data(
            num_features, samples, feature_map=feature_map
        )
        # Preserve original dtype for later use in __getitem__
        self._dtype = self.features.dtype

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        dtype = torch.float32 if self._dtype == np.float32 else torch.cfloat
        return {
            "states": torch.tensor(self.features[index], dtype=dtype),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QModel(nn.Module):
    """
    Classical feed‑forward network that can ingest either real or complex
    feature vectors.  For complex inputs the real and imaginary parts are
    concatenated along the feature dimension before passing to the linear
    layers.
    """
    def __init__(self, input_dim: int, hidden_sizes: Tuple[int,...] = (64, 32)):
        super().__init__()
        # The network internally expects double the feature dimension
        # when complex inputs are supplied.
        self.input_dim = input_dim * 2
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if state_batch.is_complex():
            # Concatenate real and imaginary parts
            state_batch = torch.cat([state_batch.real, state_batch.imag], dim=1)
        return self.net(state_batch).squeeze(-1)
