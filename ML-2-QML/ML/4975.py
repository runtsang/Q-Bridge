"""Hybrid classical QCNN.

The model stitches together:
* 2‑D convolutional layers (Quantum‑NAT style)
* A sequence of fraud‑layer modules that emulate photonic gates
* A quantum‑like fully‑connected block (linear + ReLU)
* A self‑attention head that weights the fraud output
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence
import numpy as np
import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Fraud‑layer data structure (copied from FraudDetection.py)
# --------------------------------------------------------------------------- #
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

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single fraud layer as a tiny 2‑D MLP."""
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return Layer()

def build_fraud_block(
    input_params: FraudLayerParameters,
    layers: Sequence[FraudLayerParameters],
) -> nn.Sequential:
    """Build a sequential fraud‑detection network."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# Self‑attention block (mirrors SelfAttention.py)
# --------------------------------------------------------------------------- #
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        scores = torch.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

# --------------------------------------------------------------------------- #
# Hybrid QCNN model
# --------------------------------------------------------------------------- #
class HybridQCNN(nn.Module):
    """Classical QCNN that fuses convolution, fraud layers, quantum‑like FC and self‑attention."""
    def __init__(
        self,
        fraud_input: FraudLayerParameters,
        fraud_layers: Sequence[FraudLayerParameters],
        attention_dim: int = 4,
    ) -> None:
        super().__init__()
        # Convolutional backbone (Quantum‑NAT inspired)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fraud‑layer sequence
        self.fraud_block = build_fraud_block(fraud_input, fraud_layers)
        # Quantum‑like fully‑connected block
        self.q_fc = nn.Sequential(
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        self.norm = nn.BatchNorm1d(4)
        # Self‑attention head
        self.attention = SelfAttentionBlock(embed_dim=attention_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        feats = self.features(x)
        flat = feats.view(feats.size(0), -1)
        q_out = self.q_fc(flat)
        q_out = self.norm(q_out)
        # Fraud block expects a 2‑D input; we take the first two dims
        fraud_in = q_out[:, :2]
        fraud_out = self.fraud_block(fraud_in)
        # Attention over the QCNN output and fraud output
        attn = self.attention(q_out, q_out, fraud_out)
        return attn.squeeze(-1)

__all__ = [
    "HybridQCNN",
    "FraudLayerParameters",
    "build_fraud_block",
    "SelfAttentionBlock",
]
