import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic regression data via random superposition of features."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset yielding feature vectors and continuous targets."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

class HybridQuanvolutionModel(nn.Module):
    """
    Classical hybrid model that applies a 2x2 quanvolution filter followed by a
    linear head.  The head can be a classifier or a regression network,
    selected via the `mode` argument.
    """
    def __init__(self, mode: str = "classification", num_classes: int = 10, num_features: int = 4):
        super().__init__()
        self.mode = mode
        self.qfilter = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # After conv: 4 channels, 14x14 if input is 28x28
        out_features = 4 * 14 * 14
        if self.mode == "classification":
            self.head = nn.Linear(out_features, num_classes)
        else:
            self.head = nn.Linear(out_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        features = features.view(x.size(0), -1)
        logits = self.head(features)
        if self.mode == "classification":
            return F.log_softmax(logits, dim=-1)
        else:
            return logits.squeeze(-1)

__all__ = ["HybridQuanvolutionModel", "RegressionDataset", "generate_superposition_data"]
