import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ClassicalQuanvolutionFilter(nn.Module):
    """Classical 2×2 patch extractor mimicking the quantum filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)


class QuanvolutionHybrid(nn.Module):
    """
    Hybrid model that can perform either classification or regression.
    When ``regression=True`` the final head outputs a scalar; otherwise
    it produces class logits.
    """
    def __init__(self, num_classes: int = 10, regression: bool = False) -> None:
        super().__init__()
        self.qfilter = ClassicalQuanvolutionFilter()
        hidden_dim = 4 * 14 * 14
        self.fc = nn.Linear(hidden_dim, 32)
        self.act = nn.ReLU()
        out_dim = 1 if regression else num_classes
        self.head = nn.Linear(32, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        h = self.act(self.fc(features))
        logits = self.head(h)
        if self.head.out_features == 1:
            return logits.squeeze(-1)
        return F.log_softmax(logits, dim=-1)


# --------------------------------------------------------------------------- #
# Auxiliary utilities – classical regression dataset from the quantum example
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data resembling the quantum superposition labels.
    ``x`` are uniformly sampled real vectors; ``y`` are a non‑linear
    function of their sum to mimic a quantum measurement.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding a single real feature vector and a scalar target."""
    def __init__(self, samples: int, num_features: int) -> None:
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[index], dtype=torch.float32),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }


__all__ = ["QuanvolutionHybrid", "RegressionDataset", "generate_superposition_data"]
