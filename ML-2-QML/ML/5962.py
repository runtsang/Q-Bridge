"""Classical hybrid classifier with RBF kernel and CNN backbone.

The module implements:
- RBFKernel: classical RBF kernel for similarity computations.
- CNNBackbone: convolutional feature extractor.
- DenseHead: simple dense classification head with sigmoid.
- HybridClassifier: end‑to‑end model combining backbone and head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
import numpy as np


class RBFKernel(nn.Module):
    """Classical RBF kernel."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x: (N, D), y: (M, D)
        diff = x.unsqueeze(1) - y.unsqueeze(0)          # (N, M, D)
        dist_sq = (diff ** 2).sum(-1)                   # (N, M)
        return torch.exp(-self.gamma * dist_sq)


class CNNBackbone(nn.Module):
    """Convolutional backbone identical to reference pair 2."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        # The following linear dimensions match the original seed; adjust if needed.
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DenseHead(nn.Module):
    """Dense classification head."""
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        probs = torch.sigmoid(logits + self.shift)
        return torch.cat((probs, 1 - probs), dim=-1)


class HybridClassifier(nn.Module):
    """End‑to‑end hybrid classifier using classical kernel and dense head."""
    def __init__(self, use_kernel: bool = True, gamma: float = 1.0):
        super().__init__()
        self.backbone = CNNBackbone()
        self.head = DenseHead(1)
        self.use_kernel = use_kernel
        self.kernel = RBFKernel(gamma) if use_kernel else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        if self.kernel is None:
            raise RuntimeError("Kernel not initialized.")
        return np.array([[self.kernel(a_i, b_j).item() for b_j in b] for a_i in a])
