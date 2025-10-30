import torch
from torch import nn
import numpy as np

class HybridEstimatorML(nn.Module):
    """
    Classical component of the hybrid estimator.

    It contains:
    * A standard regression head.
    * A QCNN‑inspired fully‑connected block that mimics the quantum convolutional
      and pooling stages.
    * A tiny fully‑connected layer that emulates the quantum FCL example.
    """

    def __init__(self, num_features: int = 2, qcnn_input_dim: int = 8):
        super().__init__()
        # Classical regression head
        self.regressor = nn.Sequential(
            nn.Linear(num_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        # QCNN‑inspired block (purely classical)
        self.qcnn_block = nn.Sequential(
            nn.Linear(qcnn_input_dim, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
            nn.Linear(4, 1)
        )
        # Fully connected layer mimicking the quantum FCL
        self.fcl = nn.Linear(1, 1)

    def forward(self, features: torch.Tensor,
                qcnn_features: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass that fuses classical and QCNN‑style signals.

        Parameters
        ----------
        features : Tensor (batch, num_features)
            Input for the regression head.
        qcnn_features : Tensor (batch, qcnn_input_dim) or None
            Optional features for the QCNN block.

        Returns
        -------
        Tensor (batch,)
            Hybrid prediction.
        """
        cls_out = self.regressor(features).squeeze(-1)
        if qcnn_features is not None:
            qcnn_out = self.qcnn_block(qcnn_features).squeeze(-1)
        else:
            qcnn_out = torch.zeros_like(cls_out)
        fcl_out = self.fcl(qcnn_out.unsqueeze(-1)).squeeze(-1)
        return cls_out + fcl_out


def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper that mirrors the data generation in the QuantumRegression example.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """
    Dataset that returns classical feature vectors and target values.
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


__all__ = ["HybridEstimatorML", "RegressionDataset", "generate_superposition_data"]
