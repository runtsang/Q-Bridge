"""Enhanced classical regression with self‑attention and feed‑forward head.

The model fuses a lightweight self‑attention mechanism with a deep feed‑forward
network.  The attention module learns a soft weighting of the input features,
mirroring the quantum self‑attention circuit in the companion QML implementation.
The network is fully PyTorch‑compatible and ready for standard training loops.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic dataset where each label is a non‑linear function of
    the sum of the input features.

    Parameters
    ----------
    num_features : int
        Dimensionality of the feature vector.
    samples : int
        Number of samples to generate.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Feature matrix (samples × num_features) and target vector.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Simple dataset exposing tensor dictionaries."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ClassicalSelfAttention(nn.Module):
    """
    Lightweight self‑attention module that learns a soft weighting over the
    feature dimension.  The implementation mirrors the logic of the quantum
    self‑attention circuit while remaining purely classical.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        # Rotation and entanglement parameters are learned as dense matrices.
        self.rotation = nn.Linear(input_dim, input_dim, bias=False)
        self.entangle = nn.Linear(input_dim, input_dim, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass producing an attended feature representation.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Attended representation of shape (batch, input_dim).
        """
        query = self.rotation(inputs)
        key = self.entangle(inputs)
        scores = torch.softmax(query @ key.T / np.sqrt(inputs.shape[1]), dim=-1)
        return scores @ inputs

class QModel(nn.Module):
    """
    Classical regression model that first applies self‑attention to the raw
    features and then predicts a scalar target with a small feed‑forward network.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.attention = ClassicalSelfAttention(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        state_batch : torch.Tensor
            Input batch of shape (batch, input_dim).

        Returns
        -------
        torch.Tensor
            Predicted values of shape (batch,).
        """
        attn_out = self.attention(state_batch)
        return self.net(attn_out).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_superposition_data"]
