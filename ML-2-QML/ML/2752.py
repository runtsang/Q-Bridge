import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_2d_data(kernel_size: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic 2‑D regression dataset.
    Each sample is a ``kernel_size×kernel_size`` patch of real values.
    The label is a smooth non‑linear function of the patch sum.
    """
    data = np.random.uniform(-1.0, 1.0, size=(samples, kernel_size, kernel_size)).astype(np.float32)
    angles = data.reshape(samples, -1).sum(axis=1)
    labels = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return data, labels.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset that returns a 2‑D patch and its regression target."""
    def __init__(self, samples: int, kernel_size: int):
        self.states, self.labels = generate_2d_data(kernel_size, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class ConvFilter(nn.Module):
    """
    A lightweight 2‑D convolution filter that mimics the behaviour of the
    quantum quanvolution layer. It is implemented with a single 2‑D convolution
    followed by a sigmoid non‑linearity and a global mean.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input patch of shape ``(batch, kernel_size, kernel_size)``.
        Returns
        -------
        torch.Tensor
            Global activation of shape ``(batch,)``.
        """
        x = x.unsqueeze(1)  # add channel dimension
        logits = self.conv(x)  # (batch, 1, 1, 1)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.squeeze()

class QModel(nn.Module):
    """
    Classical regression model that first extracts a single feature with
    ``ConvFilter`` and then maps it to the target with a small MLP.
    """
    def __init__(self, kernel_size: int = 2):
        super().__init__()
        self.conv = ConvFilter(kernel_size=kernel_size)
        self.mlp = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : torch.Tensor
            Batch of 2‑D patches of shape ``(batch, kernel_size, kernel_size)``.
        Returns
        -------
        torch.Tensor
            Predicted scalar target of shape ``(batch,)``.
        """
        features = self.conv(states).unsqueeze(-1)  # (batch, 1)
        return self.mlp(features).squeeze(-1)

__all__ = ["QModel", "RegressionDataset", "generate_2d_data"]
