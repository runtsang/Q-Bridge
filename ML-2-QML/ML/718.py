"""Enhanced classical regression model with optional feature extraction.

This module extends the original seed by adding:
- an optional FeatureExtractor that can be a simple MLP or a quantum circuit.
- a hybrid loss that includes a regularization term based on the variance of the predictions.
- a simple training helper method `fit` that demonstrates how to train with a DataLoader.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import Optional

def generate_superposition_data(num_features: int, samples: int, *, noise: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic regression data based on the sum of features.

    Parameters
    ----------
    num_features : int
        Dimensionality of input features.
    samples : int
        Number of samples to generate.
    noise : float, optional
        Standard deviation of Gaussian noise added to the labels.

    Returns
    -------
    features : torch.Tensor
        Shape (samples, num_features).
    labels : torch.Tensor
        Shape (samples,).
    """
    x = torch.rand(samples, num_features, dtype=torch.float32) * 2 - 1  # Uniform [-1, 1]
    angles = x.sum(dim=1)
    y = torch.sin(angles) + noise * torch.cos(2 * angles)
    return x, y

class RegressionDataset(Dataset):
    def __init__(self, samples: int, num_features: int, noise: float = 0.1):
        self.features, self.labels = generate_superposition_data(num_features, samples, noise=noise)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {"states": self.features[idx], "target": self.labels[idx]}

class FeatureExtractor(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MLPFeatureExtractor(FeatureExtractor):
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class QModel(nn.Module):
    """
    Classical regression model with optional feature extractor and hybrid loss.
    """
    def __init__(self, num_features: int, feature_extractor: Optional[FeatureExtractor] = None):
        super().__init__()
        self.feature_extractor = feature_extractor or MLPFeatureExtractor(num_features)
        # The output dimension of the feature extractor is inferred from its last layer
        out_dim = self.feature_extractor.net[-2].out_features
        self.regressor = nn.Linear(out_dim, 1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(states)
        out = self.regressor(features).squeeze(-1)
        return out

    def hybrid_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute MSE loss plus a regularization term based on the variance of predictions.
        """
        mse = F.mse_loss(preds, targets)
        var_reg = preds.var(dim=0)
        return mse + 0.01 * var_reg

    def fit(self, dataloader, epochs: int = 20, lr: float = 1e-3, device: str = "cpu"):
        """
        Simple training loop that demonstrates usage of the hybrid loss.
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                states = batch["states"].to(device)
                targets = batch["target"].to(device)
                optimizer.zero_grad()
                preds = self.forward(states)
                loss = self.hybrid_loss(preds, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.4f}")

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
