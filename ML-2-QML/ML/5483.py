"""
Classical hybrid model combining CNN, LSTM, and fraud‑style regressor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Sequence
from dataclasses import dataclass

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

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class HybridNAT(nn.Module):
    """Hybrid classical model combining CNN, LSTM, and fraud‑style regressor."""

    def __init__(self, input_channels: int = 1, hidden_dim: int = 2) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.latent_dim = 16 * 7 * 7
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        # Fraud‑style regressor
        input_params = FraudLayerParameters(
            bs_theta=0.0,
            bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        layer_params = [
            FraudLayerParameters(
                bs_theta=0.1,
                bs_phi=0.2,
                phases=(0.3, 0.4),
                squeeze_r=(0.5, 0.6),
                squeeze_phi=(0.7, 0.8),
                displacement_r=(0.9, 1.0),
                displacement_phi=(1.1, 1.2),
                kerr=(1.3, 1.4),
            )
        ]
        self.regressor = build_fraud_detection_program(input_params, layer_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, channels, H, W)
        batch, seq_len, C, H, W = x.shape
        x = x.view(batch * seq_len, C, H, W)
        feats = self.cnn(x)
        feats = feats.view(batch * seq_len, -1)
        feats = self.fc(feats)
        feats = feats.view(batch, seq_len, -1)
        lstm_out, _ = self.lstm(feats)
        last = lstm_out[:, -1, :]
        out = self.regressor(last)
        return out

__all__ = ["HybridNAT"]
