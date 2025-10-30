"""Hybrid Quanvolution network with classical and quantum-inspired layers.

The class combines a classical convolutional filter (inspired by the original
QuanvolutionFilter), a quantum-inspired fully connected layer that emulates
the expectation value of a parameterised quantum circuit, and a final
classification head.  The design allows direct comparison with the quantum
implementation while keeping the classical version lightweight.

The network can be used as a drop‑in replacement for the original
QuanvolutionClassifier in any PyTorch training loop.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Iterable

class ClassicalFC(nn.Module):
    """A lightweight quantum‑inspired fully‑connected layer.

    The layer takes a vector of parameters (thetas) and returns the mean
    of a tanh‑activated linear transformation, mimicking the expectation
    value of a simple quantum circuit.
    """
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

class QuanvolutionFilter(nn.Module):
    """Classical 2‑D convolution that reduces a 28×28 image to a 14×14
    feature map with 4 channels, matching the patch‑size used in the
    quantum version.
    """
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

class HybridQuanvolutionNet(nn.Module):
    """Hybrid network that fuses a classical quanvolution filter,
    a quantum‑inspired FC layer, and a standard classification head.

    The forward pass:
        1. Extract 2×2 patches via the convolution.
        2. Compute a scalar quantum‑inspired expectation from the
           flattened features.
        3. Concatenate the expectation with the feature vector.
        4. Classify with a linear layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.qfc = ClassicalFC(n_features=4 * 14 * 14)
        self.linear = nn.Linear(4 * 14 * 14 + 1, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)  # shape (bsz, 784)
        # Compute quantum‑inspired expectation per sample
        batch_expectation = torch.tensor(
            [self.qfc(features[i].tolist()) for i in range(features.shape[0])],
            device=features.device,
        ).view(-1, 1)  # shape (bsz, 1)
        combined = torch.cat([features, batch_expectation], dim=1)
        logits = self.linear(combined)
        return F.log_softmax(logits, dim=-1)

__all__ = ["HybridQuanvolutionNet"]
