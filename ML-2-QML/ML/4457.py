from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

class PhotonicLayer(nn.Module):
    """Classical emulation of a single photonic layer."""
    def __init__(self, params: FraudLayerParameters, clip: bool = False) -> None:
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        self.shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.linear(x))
        return y * self.scale + self.shift

class ConvFilter(nn.Module):
    """Fast classical convolution that mimics the quantum filter interface."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.view(x.size(0), -1)

class QuanvolutionFilter(nn.Module):
    """Classical emulation of the quantum convolutional filter."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class ClassicalSelfAttention(nn.Module):
    """Self‑attention block that operates on feature vectors."""
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        value = inputs
        scores = F.softmax(query @ key.T / (self.embed_dim ** 0.5), dim=-1)
        return scores @ value

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection architecture that can mix photonic, conv, and attention blocks."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        use_quantum_conv: bool = False,
        use_quantum_attention: bool = False,
    ) -> None:
        super().__init__()
        # Photonic feature extractor – kept for compatibility but not used in the forward path.
        self.photonic = PhotonicLayer(input_params, clip=False)
        # Choose between a classical conv or a quanvolution filter.
        self.conv = QuanvolutionFilter() if use_quantum_conv else ConvFilter()
        # Optional self‑attention block.
        self.attention = ClassicalSelfAttention() if use_quantum_attention else None

        # Determine feature dimensionality after the convolutional stage.
        dummy = torch.zeros(1, 1, 28, 28)
        conv_out = self.conv(dummy)
        feature_size = conv_out.size(1)
        self.classifier = nn.Linear(feature_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass the image through the chosen convolutional filter.
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        # Optional self‑attention refinement.
        if self.attention is not None:
            rot = torch.randn(self.attention.embed_dim, self.attention.embed_dim, device=y.device)
            ent = torch.randn(self.attention.embed_dim, self.attention.embed_dim, device=y.device)
            y = self.attention(rot, ent, y)
        logits = self.classifier(y)
        return torch.sigmoid(logits)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    use_quantum_conv: bool = False,
    use_quantum_attention: bool = False,
) -> FraudDetectionHybrid:
    """Return a hybrid fraud‑detection model that can mix photonic, conv, and attention blocks."""
    return FraudDetectionHybrid(
        input_params, layers, use_quantum_conv, use_quantum_attention
    )

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
]
