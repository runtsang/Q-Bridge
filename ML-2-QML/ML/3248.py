import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 stride‑2 convolution that reduces a 28×28 image
    to 14×14 patches, producing 4 feature maps."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)  # shape [B, 4*14*14]

class QuantumFullyConnectedLayer(nn.Module):
    """Surrogate for a quantum fully‑connected layer.  It emulates the
    expectation value of a parameterised two‑qubit circuit by applying a
    linear transform followed by a tanh activation and an averaging
    operation."""
    def __init__(self, n_features: int, n_outputs: int = 10) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear transform followed by tanh to mimic quantum expectation
        logits = torch.tanh(self.linear(x))
        return logits

class QuanvolutionHybrid(nn.Module):
    """Hybrid model that first applies a classical quanvolution filter
    and then passes the flattened features to a surrogate quantum
    fully‑connected layer.  The output is a 10‑dimensional log‑softmax
    suitable for MNIST classification."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.qfc = QuantumFullyConnectedLayer(n_features=4 * 14 * 14, n_outputs=10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)            # [B, 784]
        logits = self.qfc(features)           # [B, 10]
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionHybrid"]
