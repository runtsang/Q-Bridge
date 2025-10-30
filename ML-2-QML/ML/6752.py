import torch
from torch import nn
import numpy as np

class ConvGen611(nn.Module):
    """
    Hybrid classical convolution and regression model.
    Combines a Conv2d filter with a linear regression head.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 num_features: int = 8, regression: bool = True):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)
        self.regression = regression
        if regression:
            self.reg_head = nn.Sequential(
                nn.Linear(num_features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        data: Tensor of shape (batch, 1, H, W) or (H, W)
        """
        if data.dim() == 2:  # single image
            data = data.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        elif data.dim() == 3 and data.size(0) == 1:  # (1,H,W)
            data = data.unsqueeze(1)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        if self.regression:
            # flatten activations to vector
            features = activations.view(activations.size(0), -1)  # (batch, H*W)
            out = self.reg_head(features)
            return out.squeeze(-1)
        return activations.mean(dim=[2, 3]).squeeze(-1)

def generate_superposition_data(num_features: int, samples: int):
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

__all__ = ["ConvGen611", "generate_superposition_data", "RegressionDataset"]
