import torch
from torch import nn
import torch.nn.functional as F

class QCNNHybridModel(nn.Module):
    """Hybrid QCNN model combining classical convolution, quantum‑inspired layers,
    and classical regression/classification heads.
    """
    def __init__(self, input_dim: int = 8, conv_channels: int = 16) -> None:
        super().__init__()
        # Feature‑map layer inspired by the quantum ZFeatureMap
        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, conv_channels),
            nn.Tanh()
        )
        # Convolutional layers mirroring the QCNN structure
        self.conv1 = nn.Sequential(nn.Linear(conv_channels, conv_channels), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(conv_channels, conv_channels - 4), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(conv_channels - 4, conv_channels - 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(conv_channels - 8, conv_channels - 12), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(conv_channels - 12, conv_channels - 12), nn.Tanh())
        # Classical heads
        self.regressor = nn.Linear(conv_channels - 12, 1)
        self.classifier = nn.Linear(conv_channels - 12, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a regression output."""
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.regressor(x))

    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via the classifier head."""
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return F.softmax(self.classifier(x), dim=-1)

__all__ = ["QCNNHybridModel"]
