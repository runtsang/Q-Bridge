"""Hybrid classical model combining CNN-based NAT classification with a regression head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class HybridNATRegressor(nn.Module):
    """
    Combines the Quantum‑NAT convolutional backbone with an auxiliary regression head.
    The classification path mirrors the original QFCModel, while the regression path
    adds a small fully‑connected network that predicts a scalar target from the same
    image features.
    """
    def __init__(self, num_classes: int = 4, regression: bool = True):
        super().__init__()
        # Convolutional backbone (same as original QFCModel)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Flattened feature size: 16 * 7 * 7
        self.classifier = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )
        self.class_norm = nn.BatchNorm1d(num_classes)

        self.regression = regression
        if regression:
            # Regression head operating on the same flattened features
            self.regressor = nn.Sequential(
                nn.Linear(16 * 7 * 7, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )
            self.reg_norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Returns a dictionary with keys 'class' and optionally'regress'.
        """
        bsz = x.shape[0]
        feats = self.features(x)
        flat = feats.view(bsz, -1)
        logits = self.classifier(flat)
        out = {"class": self.class_norm(logits)}
        if self.regression:
            reg = self.regressor(flat)
            out["regress"] = self.reg_norm(reg).squeeze(-1)
        return out

# Data utilities --------------------------------------------------------------

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic data for the regression task.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Dataset wrapping the synthetic superposition data.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }

__all__ = ["HybridNATRegressor", "RegressionDataset", "generate_superposition_data"]
