import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class HybridQCNN(nn.Module):
    """Hybrid classical QCNN model with convolutional backbone and graph‑adjacency output."""
    def __init__(self, in_channels: int = 1, num_classes: int = 4):
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected projection
        self.fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            out: Normalized class logits.
            adj: Pairwise cosine‑similarity adjacency matrix (binary thresholded).
        """
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        out = self.fc(flattened)
        out = self.norm(out)

        # Pairwise cosine similarity as graph adjacency
        norm = out / (out.norm(dim=1, keepdim=True) + 1e-12)
        sim = torch.mm(norm, norm.t())
        threshold = 0.8
        adj = (sim > threshold).float()
        return out, adj
