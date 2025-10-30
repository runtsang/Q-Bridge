"""Enhanced classical QCNN: residual blocks and adaptive pooling."""
import torch
from torch import nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """1‑D residual block with skip connection."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.act = nn.Tanh()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.proj(x) + x)

class AdaptivePool(nn.Module):
    """Learnable interpolation between mean and max pooling."""
    def __init__(self, dim: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.dim = dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=self.dim, keepdim=True)
        max_val = x.max(dim=self.dim, keepdim=True)[0]
        return self.alpha * mean + (1 - self.alpha) * max_val

class QCNNEnhanced(nn.Module):
    """Deeper QCNN‑style network with residuals and adaptive pooling."""
    def __init__(self, feature_dim: int = 8, num_features: int = 4):
        super().__init__()
        # Feature map depth is learnable via additional linear layers
        self.feature_map = nn.Sequential(
            nn.Linear(feature_dim, 16), nn.Tanh(),
            nn.Linear(16, 32), nn.Tanh()
        )
        self.conv1 = ResidualBlock(32, 32)
        self.pool1 = AdaptivePool(1)
        self.conv2 = ResidualBlock(32, 16)
        self.pool2 = AdaptivePool(1)
        self.conv3 = ResidualBlock(32, 8)
        self.head = nn.Linear(8, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))
