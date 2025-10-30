"""Hybrid classical model combining CNN, Conv filter and fraud‑detection style layers."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, List, Optional

# ----------------------------------------------------------------------
# Conv filter (adapted from Conv.py)
# ----------------------------------------------------------------------
def Conv(kernel_size: int = 2, threshold: float = 0.0) -> nn.Module:
    """Return a lightweight 2‑D convolution filter."""
    class ConvFilter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data: torch.Tensor) -> float:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            return activations.mean().item()

    return ConvFilter()
# ----------------------------------------------------------------------
# Fraud‑layer parameters (adapted from FraudDetection.py)
# ----------------------------------------------------------------------
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

def _layer_from_params(
    params: FraudLayerParameters,
    *,
    clip: bool = False,
) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            return outputs * self.scale + self.shift

    return Layer()
# ----------------------------------------------------------------------
# Main hybrid network
# ----------------------------------------------------------------------
class QuantumNATHybrid(nn.Module):
    """Classical CNN + fraud‑detection style FC stack with optional Conv filter."""
    def __init__(
        self,
        conv_kernel: int = 2,
        fraud_params: Optional[List[FraudLayerParameters]] = None,
    ) -> None:
        super().__init__()
        # Feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Conv filter for optional preprocessing
        self.conv_filter = Conv(kernel_size=conv_kernel)
        # Fraud‑style layers
        self.fraud_layers = nn.Sequential(
            *(_layer_from_params(p, clip=False) for p in fraud_params or [])
        )
        # Final classifier
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Feature extraction
        x = self.features(x)
        # Optional Conv filter on a single patch (example usage)
        # patch = x[:, :, :self.conv_filter.kernel_size, :self.conv_filter.kernel_size]
        # _ = self.conv_filter.run(patch.detach().cpu().numpy())
        x = x.view(x.size(0), -1)
        # Fraud‑style processing
        x = self.fraud_layers(x) if self.fraud_layers else x
        # Classification head
        x = self.fc(x)
        return self.norm(x)

__all__ = ["QuantumNATHybrid", "FraudLayerParameters"]
