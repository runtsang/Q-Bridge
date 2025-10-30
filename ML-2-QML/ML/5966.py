import torch
import torch.nn as nn

__all__ = ["QuantumClassifierModel"]

class QuantumClassifierModel(nn.Module):
    """
    Classical hybrid classifier.

    Architecture:
        - 2‑layer CNN (8 and 16 filters) with 2×2 pooling.
        - Flattened feature map fed to a depth‑controlled fully‑connected
          stack mirroring the quantum ansatz.
        - Final linear layer outputs logits for two classes.
        - Batch‑norm on the classifier output for stability.
    """
    def __init__(self, num_features: int = 64, depth: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        dummy_input = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            dummy_out = self.features(dummy_input)
        flattened_dim = dummy_out.view(1, -1).shape[1]
        layers = []
        in_dim = flattened_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, num_features))
            layers.append(nn.ReLU(inplace=True))
            in_dim = num_features
        layers.append(nn.Linear(in_dim, 2))
        self.head = nn.Sequential(*layers)
        self.norm = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        flattened = features.view(features.size(0), -1)
        logits = self.head(flattened)
        return self.norm(logits)
