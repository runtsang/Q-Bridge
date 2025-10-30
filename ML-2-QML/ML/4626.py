import torch
from torch import nn
import torch.nn.functional as F

class QuanvolutionFilter(nn.Module):
    """Apply a 2×2 kernel to produce 4 features per patch, mimicking a quantum filter."""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class EstimatorQNN(nn.Module):
    """Hybrid classical network that combines quanvolution, CNN, and regression head."""
    def __init__(self):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.features = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 1))
        self.norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        x = self.qfilter(x)  # (bsz, 4*14*14)
        x = x.unsqueeze(1)   # add channel dim for conv layers
        x = self.features(x)
        x = x.view(bsz, -1)
        x = self.fc(x)
        return self.norm(x)
