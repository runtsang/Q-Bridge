import torch
from torch import nn
import numpy as np

class QCNNRegressionHybrid(nn.Module):
    """
    Classical hybrid model that mimics a QCNN using convolutional layers and a regression head.
    The architecture consists of an initial linear feature map, followed by 1‑D convolutions,
    pooling (implemented via additional linear layers), and a final regression head.
    """
    def __init__(self, num_features: int, num_wires: int):
        """
        Parameters
        ----------
        num_features
            Dimensionality of the input feature vector.
        num_wires
            Number of qubits that the quantum part would use.  It is used only for
            compatibility with the quantum module and is not part of the classical
            computation.
        """
        super().__init__()
        self.num_wires = num_wires

        # Feature extractor: linear -> ReLU -> Conv1d -> ReLU -> Conv1d -> ReLU
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Flatten and fully connected regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x
            Tensor of shape (batch_size, num_features).

        Returns
        -------
        torch.Tensor
            Regression output of shape (batch_size,).
        """
        # Expand dimension for Conv1d: (batch, 1, features)
        x = x.unsqueeze(1)
        features = self.feature_extractor(x)
        out = self.regressor(features)
        return out.squeeze(-1)

def generate_superposition_data(num_features: int, samples: int):
    """
    Generate simple regression data: y = sin(sum(x)) + 0.1 * cos(2 * sum(x)).
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns a dictionary with keys 'features' and 'target'.
    """
    def __init__(self, samples: int, num_features: int):
        self.features, self.targets = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int):
        return {
            "features": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.targets[idx], dtype=torch.float32)
        }

__all__ = ["QCNNRegressionHybrid", "RegressionDataset", "generate_superposition_data"]
