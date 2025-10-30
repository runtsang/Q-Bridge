"""UnifiedQCNN: a hybrid classical‑quantum convolutional architecture for regression & classification.

This module implements a classical neural network that mirrors the
quantum QCNN structure described in the seed.  It can be used for
regression or classification and is fully compatible with the
training pipelines that expect a ``QCNN`` factory.

The class name ``UnifiedQCNN`` is deliberately chosen to be the same
as the quantum counterpart defined in ``qml_code`` so that the same
API can be swapped at runtime.
"""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset

# ----- Utility data generation -----
def generate_superposition_data(num_features: int, samples: int):
    """Generate synthetic regression data."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)

class RegressionDataset(Dataset):
    """Dataset for quantum regression demo."""
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {"states": torch.tensor(self.features[idx], dtype=torch.float32),
                "target": torch.tensor(self.labels[idx], dtype=torch.float32)}

# ----- Classical Conv filter (drop‑in for quanvolution) -----
class ConvFilter(nn.Module):
    """Simple 2‑D convolutional filter."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the filter on a 2‑D input."""
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

# ----- Classical classifier helper -----
def build_classifier_circuit(num_features: int, depth: int):
    """Return a feed‑forward classifier and metadata."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# ----- Unified QCNN model -----
class UnifiedQCNN(nn.Module):
    """Classical QCNN‑style network.

    Parameters
    ----------
    num_features : int
        Size of the input feature vector.
    depth : int
        Number of convolution/pooling blocks.
    head_dim : int
        Output dimension (1 for regression, 2 for binary classification).
    """
    def __init__(self, num_features: int = 8, depth: int = 3, head_dim: int = 1):
        super().__init__()
        # Feature map
        self.feature_map = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.Tanh()
        )
        # Convolution & pooling blocks
        self.blocks = nn.ModuleList()
        in_dim = 16
        for _ in range(depth):
            conv = nn.Sequential(
                nn.Linear(in_dim, 16),
                nn.Tanh()
            )
            pool = nn.Sequential(
                nn.Linear(16, max(4, in_dim // 2)),
                nn.Tanh()
            )
            self.blocks.append(nn.Sequential(conv, pool))
            in_dim = pool[-1].out_features
        # Head
        self.head = nn.Linear(in_dim, head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        if self.head.out_features == 1:
            return torch.sigmoid(logits)
        return logits

def QCNN() -> UnifiedQCNN:
    """Factory returning a default classical QCNN."""
    return UnifiedQCNN()

__all__ = ["UnifiedQCNN", "QCNN", "ConvFilter",
           "RegressionDataset", "build_classifier_circuit"]
