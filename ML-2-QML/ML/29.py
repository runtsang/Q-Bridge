import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a synthetic regression dataset where each sample is a vector of
    `num_features` real values in [-1, 1] and the label is a non‑linear
    combination of the feature sums.  The function is intentionally
    richer than the seed to give the model more structure to learn.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    # Non‑linear mapping with higher‑order terms
    y = np.sin(angles) + 0.3 * np.cos(2 * angles) + 0.05 * np.sin(4 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """
    Torch dataset that returns a dictionary with keys ``states`` and ``target``.
    The ``states`` tensor contains the feature vector and ``target`` holds the
    corresponding regression target.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class QuantumRegression(nn.Module):
    """
    Classical neural network with optional quantum‑inspired feature mapping.
    The architecture is a small fully‑connected network with batch‑norm,
    ReLU activations and dropout.  A `use_qmap` flag applies a
    sin/cos transformation to each feature, mimicking a simple feature
    map that a quantum circuit might implement.
    """
    def __init__(
        self,
        num_features: int,
        hidden_sizes: list[int] | tuple[int,...] = (64, 32),
        dropout: float = 0.1,
        use_qmap: bool = False,
    ):
        super().__init__()
        self.use_qmap = use_qmap
        layers = []
        in_features = num_features * (2 if use_qmap else 1)
        for h in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(in_features, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_features = h
        layers.append(nn.Linear(in_features, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.  If `use_qmap` is True,
        the input is transformed by concatenating sin and cos of each
        feature before being fed to the linear layers.
        """
        if self.use_qmap:
            sin_part = torch.sin(state_batch)
            cos_part = torch.cos(state_batch)
            state_batch = torch.cat([sin_part, cos_part], dim=-1)
        return self.net(state_batch).squeeze(-1)

__all__ = ["QuantumRegression", "RegressionDataset", "generate_superposition_data"]
