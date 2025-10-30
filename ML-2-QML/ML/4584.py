"""SelfAttentionHybrid: classical implementation with regression/classification support.

The module preserves the original SelfAttention interface while expanding it to
support both regression and classification tasks.  It also brings in dataset
helpers from the QuantumRegression and QuantumClassifierModel references,
allowing users to build experiments without leaving the classical ecosystem.

Author: GPT‑OSS‑20B
"""

from __future__ import annotations

import math
from typing import Literal, Tuple, Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------------------------------------------------------------------- #
# 1. Dataset helpers (regression + classification)                           #
# --------------------------------------------------------------------------- #

def generate_superposition_data(
    num_features: int,
    samples: int,
    *,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the same distribution used in QuantumRegression.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of samples to generate.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray shape (samples, num_features)
        Uniformly distributed features.
    y : np.ndarray shape (samples,)
        Target values computed as sin(sum(x)) + 0.1 * cos(2 * sum(x)).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = X.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return X, y.astype(np.float32)


class RegressionDataset(Dataset):
    """Dataset wrapping the superposition data for regression."""

    def __init__(self, samples: int, num_features: int, *, seed: int | None = None):
        self.features, self.labels = generate_superposition_data(num_features, samples, seed=seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


def generate_classification_data(
    num_features: int,
    samples: int,
    *,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple 2‑class linear separator for quick experiments.
    The label is 1 if the dot product with a random unit vector exceeds 0, else 0.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(samples, num_features)).astype(np.float32)
    w = rng.normal(size=(num_features,)).astype(np.float32)
    w /= np.linalg.norm(w)
    logits = X @ w
    y = (logits > 0).astype(np.float32)
    return X, y.astype(np.float32)


class ClassificationDataset(Dataset):
    """Dataset wrapping the synthetic classification data."""

    def __init__(self, samples: int, num_features: int, *, seed: int | None = None):
        self.features, self.labels = generate_classification_data(num_features, samples, seed=seed)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


# --------------------------------------------------------------------------- #
# 2. Classical self‑attention with optional regression/classification head  #
# --------------------------------------------------------------------------- #

class ClassicalSelfAttentionHybrid:
    """Classical self‑attention block that can feed a regression or classification head.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the input embeddings.
    task : Literal['regression', 'classification']
        Which downstream head to attach.
    hidden_dim : int, optional
        Size of the hidden layer for the head (default: 32).
    """

    def __init__(
        self,
        embed_dim: int,
        task: Literal["regression", "classification"],
        hidden_dim: int = 32,
    ):
        self.embed_dim = embed_dim
        self.task = task

        # Core attention linear layers
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

        # Optional head
        if task == "regression":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif task == "classification":
            self.head = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2),
            )
        else:
            raise ValueError("task must be'regression' or 'classification'")

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: np.ndarray | None = None,
        entangle_params: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Compute self‑attention and optionally feed the head.

        Parameters
        ----------
        inputs : torch.Tensor shape (batch, embed_dim)
            Input embeddings.
        rotation_params : np.ndarray, optional
            Accepted for API compatibility; ignored in the classical path.
        entangle_params : np.ndarray, optional
            Accepted for API compatibility; ignored in the classical path.
        """
        # Compute query/key/value
        q = self.query(inputs)
        k = self.key(inputs)
        v = self.value(inputs)

        # Attention scores
        scores = torch.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.embed_dim), dim=-1
        )
        attn_output = torch.matmul(scores, v)

        # Feed through head
        return self.head(attn_output)


# --------------------------------------------------------------------------- #
# 3. Factory helpers (mirroring the quantum side)                               #
# --------------------------------------------------------------------------- #

def build_classifier_circuit(
    num_features: int,
    depth: int,
) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    """Return a feed‑forward classifier with metadata.

    This mirrors the quantum `build_classifier_circuit` signature so that a user can
    instantiate a classical classifier using identical parameters.  The function
    is kept lightweight to avoid pulling in heavy libraries when only the
    classical side is needed.

    Returns
    -------
    network : nn.Module
        Sequential classifier.
    encoding : list[int]
        Indices of the feature dimensions used as encoding.
    weight_sizes : list[int]
        Number of trainable parameters per layer.
    observables : list[int]
        Placeholder for observables; in the classical context these are simply
        the output neuron indices.
    """
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables


__all__ = [
    "ClassicalSelfAttentionHybrid",
    "RegressionDataset",
    "ClassificationDataset",
    "build_classifier_circuit",
]
