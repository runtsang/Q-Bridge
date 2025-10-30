"""Hybrid Classical Classification architecture with multiple feature extractors."""

from __future__ import annotations

from typing import Iterable, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------- #
# Residual MLP backbone (from QuantumClassifierModel.py)
# --------------------------------------------------------------------------- #
class ResidualMLP(nn.Module):
    """A scalable MLP that mirrors the original `build_classifier_circuit` but
    with residual skip connections.  The depth argument controls the number of
    hidden blocks; each block comprises a linear layer followed by ReLU and a
    residual addition to the input of that block."""
    def __init__(self, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks: List[nn.Module] = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ))
        self.output_head = nn.Linear(hidden_dim, 2)  # binary by default

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for blk in self.blocks:
            h = h + blk(h)  # residual
        return self.output_head(h)

# --------------------------------------------------------------------------- #
# Conv filter (from Conv.py)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Return a callable object that emulates the quantum filter with PyTorch ops."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        tensor = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

# --------------------------------------------------------------------------- #
# Quanvolution filter (from Quanvolution.py)
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Classical quanvolution filter that extracts 2×2 patches and flattens."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

# --------------------------------------------------------------------------- #
# Fraud detection layer (from FraudDetection.py)
# --------------------------------------------------------------------------- #
class FraudLayer(nn.Module):
    """Custom layer built from photonic parameters."""
    def __init__(self, params: dict, clip: bool = True) -> None:
        super().__init__()
        weight = torch.tensor([[params["bs_theta"], params["bs_phi"]],
                               [params["squeeze_r"][0], params["squeeze_r"][1]]],
                              dtype=torch.float32)
        bias = torch.tensor(params["phases"], dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = torch.tensor(params["displacement_r"], dtype=torch.float32)
        self.shift = torch.tensor(params["displacement_phi"], dtype=torch.float32)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.activation(self.linear(inputs))
        outputs = outputs * self.scale + self.shift
        return outputs

# --------------------------------------------------------------------------- #
# Hybrid classifier (combines any of the above modules)
# --------------------------------------------------------------------------- #
class HybridQuantumClassifier(nn.Module):
    """
    A single entry‑point that stitches together the classical backbone
    and optional feature extractors.  The constructor accepts any
    nn.Module as `backbone`; if None, a ResidualMLP is used by default.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int = 64,
                 depth: int = 3,
                 backbone: nn.Module = None,
                 **kwargs) -> None:
        super().__init__()
        self.backbone = backbone or ResidualMLP(input_dim, hidden_dim, depth)
        # Final classification head
        if hasattr(self.backbone, "output_head"):
            in_features = self.backbone.output_head.out_features
        else:
            in_features = hidden_dim
        self.classifier = nn.Linear(in_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits

    def feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """Return the intermediate representation before the final head."""
        return self.backbone(x)
