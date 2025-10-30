import torch
from torch import nn
import torch.nn.functional as F

class HybridQCNN(nn.Module):
    """Hybrid quantum‑classical convolution network.

    The architecture merges ideas from:
    * QCNN.py – classical fully connected stack mimicking quantum conv
    * QuantumNAT.py – quantum fully‑connected block with random layers
    * Quanvolution.py – two‑qubit quantum kernel for local patches
    * QCNN.py (quantum part) – convolution, pooling, and feature map

    The network consists of a shallow CNN that outputs a 4‑dimensional
    embedding, followed by a linear head.  This classical part can be
    trained independently or jointly with the quantum counterpart.

    """
    def __init__(self, in_channels=1, img_size=28, out_features=1):
        super().__init__()
        # Classical feature extractor – 2‑layer CNN
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 14×14
            nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 7×7
        )
        # Projection to a 4‑dimensional latent space
        self.fc = nn.Sequential(
            nn.Linear(4 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        # Normalisation and final head
        self.norm = nn.BatchNorm1d(4)
        self.head = nn.Linear(4, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        feat = self.features(x)
        flat = feat.view(feat.size(0), -1)
        latent = self.fc(flat)
        normed = self.norm(latent)
        return self.head(normed)

__all__ = ["HybridQCNN"]
