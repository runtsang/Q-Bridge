import torch
from torch import nn
import torch.nn.functional as F

class HybridQCNNModel(nn.Module):
    """Hybrid QCNN with classical conv‑pool layers, a sampling head and a classification head."""

    def __init__(self, input_dim: int = 8, num_classes: int = 1):
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        # Convolutional modules
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        # Sampler head (softmax over 2 outputs)
        self.sampler = nn.Sequential(nn.Linear(4, 2), nn.Tanh())
        # Classification head
        self.head = nn.Linear(2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        samp = self.sampler(x)
        logits = self.head(samp)
        return torch.sigmoid(logits)

def HybridQCNN() -> HybridQCNNModel:
    """Factory returning the configured :class:`HybridQCNNModel`."""
    return HybridQCNNModel()
