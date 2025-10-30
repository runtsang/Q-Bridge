"""Hybrid classical model inspired by Quantum‑NAT, fraud detection, and classifier seeds."""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class FraudLayerParameters:
    """Parameters for the fraud‑detection inspired linear block."""
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

class FraudLayer(nn.Module):
    """Parameterized linear + tanh + scaling/shift block mirroring photonic layers."""
    def __init__(self, params: FraudLayerParameters, clip: bool = False) -> None:
        super().__init__()
        weight = torch.tensor(
            [
                [params.bs_theta, params.bs_phi],
                [params.squeeze_r[0], params.squeeze_r[1]],
            ],
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
        self.linear = linear
        self.activation = activation
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        y = self.activation(self.linear(x))
        return y * self.scale + self.shift

class QuantumNATHybrid(nn.Module):
    """Hybrid classical model combining CNN, fraud‑detection block, and deep classifier."""
    def __init__(self,
                 fraud_params: FraudLayerParameters,
                 classifier_depth: int = 3,
                 num_features: int = 4,
                 ) -> None:
        super().__init__()
        # Feature extractor (CNN)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fraud‑detection inspired block
        self.fraud_block = FraudLayer(fraud_params, clip=True)
        # Classifier head
        in_dim = self._feature_dim() + 2  # flattened features + fraud output
        self.classifier = self._build_classifier(in_dim, num_features, classifier_depth)
        self.norm = nn.BatchNorm1d(2)

    def _feature_dim(self) -> int:
        # After two 2x2 max pools on 28x28 input: 28/2/2 = 7
        return 16 * 7 * 7

    def _build_classifier(self, in_dim: int, num_features: int, depth: int) -> nn.Sequential:
        layers = []
        in_dim_curr = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_dim_curr, num_features))
            layers.append(nn.ReLU())
            in_dim_curr = num_features
        layers.append(nn.Linear(in_dim_curr, 2))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Feature extraction
        f = self.features(x)
        flat = f.view(f.shape[0], -1)
        # Extract two channel means for the fraud block
        channel_means = f.mean(dim=(2, 3))[:, :2]
        fraud_out = self.fraud_block(channel_means)
        # Concatenate and classify
        concat = torch.cat([flat, fraud_out], dim=1)
        logits = self.classifier(concat)
        return self.norm(logits)

__all__ = ["QuantumNATHybrid", "FraudLayerParameters", "FraudLayer"]
