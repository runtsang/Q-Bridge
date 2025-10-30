import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable, Sequence

class HybridFCL(nn.Module):
    """
    Hybrid classical model that merges a lightweight CNN backbone,
    a fully‑connected projection, and an optional quantum‑inspired head.
    """
    def __init__(self, in_channels: int=3, num_classes: int=4, use_quantum_head: bool=False):
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Compute flattened feature size
        dummy = torch.zeros(1, in_channels, 32, 32)
        flat_size = self.features(dummy).view(1, -1).size(1)
        # Fully connected projection
        self.fc = nn.Sequential(
            nn.Linear(flat_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.norm = nn.BatchNorm1d(num_classes)
        self.use_quantum_head = use_quantum_head
        if use_quantum_head:
            # Quantum‑inspired head: simple parameterized linear + tanh
            self.quantum_head = nn.Linear(num_classes, 1)
            nn.init.uniform_(self.quantum_head.weight, -0.1, 0.1)
            nn.init.constant_(self.quantum_head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        flat = feats.view(feats.size(0), -1)
        out = self.fc(flat)
        out = self.norm(out)
        if self.use_quantum_head:
            out = torch.tanh(self.quantum_head(out))
        return out

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
        """
        Classical RBF kernel computed on the model output embeddings.
        """
        out_a = self.forward(torch.stack(a))
        out_b = self.forward(torch.stack(b))
        mat = np.zeros((len(a), len(b)))
        for i, va in enumerate(out_a.detach().cpu().numpy()):
            for j, vb in enumerate(out_b.detach().cpu().numpy()):
                mat[i, j] = np.exp(-gamma * np.sum((va - vb) ** 2))
        return mat

__all__ = ["HybridFCL"]
