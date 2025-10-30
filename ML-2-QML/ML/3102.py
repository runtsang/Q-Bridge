import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression data that mimics a quantum superposition
    pattern.  The labels are a non‑linear combination of the input angles.
    """
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class ConvFilter(nn.Module):
    """
    Classical convolutional filter that emulates the behaviour of a quantum
    quanvolution layer.  It is a drop‑in replacement that can be stacked
    before any linear head.
    """
    def __init__(self, kernel_size: int = 1, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, num_features).  For a 2‑D filter the input must
            be reshaped to (batch, 1, kernel, kernel).

        Returns
        -------
        torch.Tensor
            Activated output averaged over spatial dimensions,
            shape (batch, 1).
        """
        x = x.view(-1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])

class RegressionDataset(Dataset):
    """
    Dataset that optionally applies a convolutional filter to the
    generated features before returning them.
    """
    def __init__(self, samples: int, num_features: int,
                 conv_filter: ConvFilter | None = None) -> None:
        self.features, self.labels = generate_superposition_data(num_features, samples)
        self.conv_filter = conv_filter

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        states = torch.tensor(self.features[index], dtype=torch.float32)
        if self.conv_filter is not None:
            # Apply the classical convolutional filter as a preprocessing step
            states = self.conv_filter(states)
        return {"states": states.squeeze(-1), "target": torch.tensor(self.labels[index], dtype=torch.float32)}

class HybridRegressionModel(nn.Module):
    """
    Classical regression model that can optionally prepend a convolutional
    filter before feeding the data into a small fully‑connected network.
    """
    def __init__(self, num_features: int, use_conv: bool = False,
                 kernel_size: int = 1, threshold: float = 0.0) -> None:
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            if num_features!= kernel_size ** 2:
                raise ValueError("num_features must equal kernel_size**2 when using conv.")
            self.conv = ConvFilter(kernel_size=kernel_size, threshold=threshold)
            input_dim = 1  # ConvFilter outputs a single scalar per sample
        else:
            self.conv = None
            input_dim = num_features

        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, state_batch: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.use_conv:
            # state_batch shape (batch, num_features) -> (batch, 1)
            x = self.conv(state_batch)
            x = x.squeeze(-1)
        else:
            x = state_batch
        return self.net(x).squeeze(-1)

__all__ = ["HybridRegressionModel", "RegressionDataset", "generate_superposition_data"]
