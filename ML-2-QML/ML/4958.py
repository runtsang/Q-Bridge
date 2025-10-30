"""Classical hybrid network that combines a convolutional filter, a Fraud‑Detection style stack, and a fully‑connected head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List


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


class FraudLayer(nn.Module):
    """Classical surrogate of a photonic fraud‑detection layer."""
    def __init__(self, params: FraudLayerParameters, clip: bool = True):
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
        # Apply the layer pair‑wise to the input.
        batch, n = x.shape
        if n % 2!= 0:
            # Pad with zeros if odd.
            x = torch.cat([x, torch.zeros(batch, 1, device=x.device)], dim=1)
            n += 1
        x_pair = x.view(batch, n // 2, 2)
        out = self.activation(self.linear(x_pair))
        out = out * self.scale + self.shift
        return out.view(batch, -1)


class Quanvolution__gen185(nn.Module):
    """Classical hybrid network: 2×2 conv → fraud‑style layers → fully‑connected head."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        fraud_layers: List[FraudLayerParameters],
        hidden_dim: int = 32,
        num_classes: int = 10,
    ):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # Fraud‑style stack: first layer un‑clipped, subsequent layers clipped.
        self.fraud_modules = nn.ModuleList(
            [FraudLayer(input_params, clip=False)] +
            [FraudLayer(p, clip=True) for p in fraud_layers]
        )
        self.fc = nn.Sequential(
            nn.Linear(4 * 14 * 14, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x).view(x.size(0), -1)
        out = features
        for layer in self.fraud_modules:
            out = layer(out)
        logits = self.fc(out)
        return F.log_softmax(logits, dim=-1)


__all__ = ["FraudLayerParameters", "FraudLayer", "Quanvolution__gen185"]
