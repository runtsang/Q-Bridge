import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class HybridNATModel(nn.Module):
    """
    Classical hybrid model that emulates the structure of the Quantum‑NAT
    seed while adding a quantum‑style random linear transformation.

    Architecture:
        - 2‑layer CNN backbone (Conv → ReLU → MaxPool → Conv → ReLU → MaxPool)
        - Fixed random linear layer (mimicking the RandomLayer in the quantum
          implementation) – weights are sampled once and frozen.
        - Trainable variational head (Linear → ReLU → Linear) producing 4 outputs.
        - BatchNorm1d on the final 4‑dimensional vector.
    """

    def __init__(self) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # Random linear layer (fixed, non‑trainable) to emulate quantum random layer
        in_features = 16 * 7 * 7  # 28x28 input → 7x7 after two 2×2 poolings
        self.random_linear = nn.Linear(in_features, 64, bias=False)
        nn.init.normal_(self.random_linear.weight, mean=0.0, std=0.1)
        self.random_linear.weight.requires_grad = False
        # Trainable variational head
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4),
        )
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        features = self.features(x)
        flat = features.view(bsz, -1)
        rand = self.random_linear(flat)
        out = self.head(rand)
        return self.norm(out)


__all__ = ["HybridNATModel"]
