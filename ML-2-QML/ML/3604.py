import torch
from torch import nn
import numpy as np

class HybridFCQCNN(nn.Module):
    """
    Classical hybrid network combining:
      * an initial parameterized fully connected layer (FCL) that learns a scalar weight.
      * a convolution‑inspired feature extractor mirroring the QCNN architecture.
      * a final linear head for regression/classification.
    The design is deliberately lightweight so that it can be substituted for a quantum layer during ablation studies.
    """
    def __init__(self, n_input: int = 8) -> None:
        super().__init__()
        # FCL‑style linear that learns a single scalar weight
        self.fcl = nn.Linear(1, 1, bias=False)
        # QCNN‑style feature extractor
        self.feature_map = nn.Sequential(nn.Linear(n_input, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Final head
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # Apply the lightweight FCL mapping to the first feature
        fcl_out = torch.tanh(self.fcl(x[:, :1]))
        # Concatenate the processed feature with the remaining ones
        x = torch.cat([fcl_out, x[:, 1:]], dim=1)
        return torch.sigmoid(self.head(x))

def HybridFCQCNNFactory() -> HybridFCQCNN:
    """Factory returning a fully‑configured instance."""
    return HybridFCQCNN()

__all__ = ["HybridFCQCNN", "HybridFCQCNNFactory"]
