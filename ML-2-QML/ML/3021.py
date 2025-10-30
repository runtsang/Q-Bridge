"""Hybrid regression/classification dataset and model for classical experiments, extending the original QuantumRegression seed."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import Iterable, Tuple

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample states of the form cos(theta)|0..0> + e^{i phi} sin(theta)|1..1>.
    Returns features, regression labels and binary classification labels.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y_reg = np.sin(angles) + 0.1 * np.cos(2 * angles)
    y_cls = (y_reg >= 0).astype(np.float32)
    return x, y_reg.astype(np.float32), y_cls

class HybridDataset(Dataset):
    """
    Dataset providing state vectors, regression targets and binary classification targets.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.regression_targets, self.classification_targets = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "regression_target": torch.tensor(self.regression_targets[index], dtype=torch.float32),
            "classification_target": torch.tensor(self.classification_targets[index], dtype=torch.float32),
        }

class HybridModel(nn.Module):
    """
    Classical neural network with shared feature extractor and two heads: regression and classification.
    """
    def __init__(self, num_features: int, hidden_sizes: Iterable[int] = (32, 16)):
        super().__init__()
        layers = []
        in_dim = num_features
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.feature_extractor = nn.Sequential(*layers)
        self.regression_head = nn.Linear(in_dim, 1)
        self.classification_head = nn.Linear(in_dim, 2)

    def forward(self, state_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(state_batch)
        regression_output = self.regression_head(features).squeeze(-1)
        classification_output = self.classification_head(features)
        return regression_output, classification_output

# Backwards‑compatibility aliases
RegressionDataset = HybridDataset
QModel = HybridModel

__all__ = ["HybridModel", "HybridDataset", "RegressionDataset", "QModel", "generate_superposition_data"]
