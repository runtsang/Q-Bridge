import torch
from torch import nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Classical approximation of the quantum filter used in the original quanvolution example."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)          # (batch, 4, 14, 14)
        return features.view(x.size(0), -1)  # flatten to (batch, 4*14*14)

class QCNNModel(nn.Module):
    """Classical network that mirrors the quantum convolution structure."""
    def __init__(self, input_dim: int = 4 * 14 * 14):
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 128), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(128, 128), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(128, 96), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(96, 64), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(64, 32), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(32, 32), nn.Tanh())
        self.head = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class QCNNHybrid(nn.Module):
    """Hybrid QCNN that first applies a quanvolution filter then a quantum‑inspired FC stack."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.qcnn = QCNNModel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        return self.qcnn(features)

__all__ = ["QCNNHybrid"]
