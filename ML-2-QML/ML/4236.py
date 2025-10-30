"""
Hybrid classical model that fuses a convolution, self‑attention, and fraud‑detection.
Author: gpt-oss-20b
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, List

# --------------------------------------------------------------------------- #
# 1. Convolutional filter --------------------------------------------------- #
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """
    2‑D convolution followed by a sigmoid activation and batch‑wise averaging.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        # Reduce to a single scalar per sample
        return activations.mean(dim=[1, 2, 3])


# --------------------------------------------------------------------------- #
# 2. Self‑attention block --------------------------------------------------- #
# --------------------------------------------------------------------------- #
class ClassicalSelfAttention(nn.Module):
    """
    Scaled dot‑product self‑attention operating on a batch of embeddings.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, embed_dim, bias=False)
        self.key   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        scores = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.bmm(scores, v)


# --------------------------------------------------------------------------- #
# 3. Fraud‑detection layer (classical) ------------------------------------ #
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


class FraudLayer(nn.Module):
    """
    A single photonic‑style layer implemented in PyTorch.
    """
    def __init__(self, params: FraudLayerParameters):
        super().__init__()
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
        bias   = torch.tensor(params.phases, dtype=torch.float32)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.scale = torch.tensor(params.displacement_r, dtype=torch.float32)
        self.shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out


class FraudDetectionModel(nn.Module):
    """
    Sequential fraud‑detection network mirroring the photonic architecture.
    """
    def __init__(self, input_params: FraudLayerParameters,
                 layer_params: Iterable[FraudLayerParameters]):
        super().__init__()
        layers: List[nn.Module] = [FraudLayer(input_params)]
        layers.extend(FraudLayer(p) for p in layer_params)
        self.layers = nn.Sequential(*layers, nn.Linear(2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# --------------------------------------------------------------------------- #
# 4. Hybrid pipeline -------------------------------------------------------- #
# --------------------------------------------------------------------------- #
class HybridConvAttentionFraud(nn.Module):
    """
    End‑to‑end classical model: conv → attention → fraud detection.
    """
    def __init__(self,
                 conv_kernel: int = 2,
                 conv_thresh: float = 0.0,
                 embed_dim: int = 4,
                 fraud_input_params: FraudLayerParameters | None = None,
                 fraud_layer_params: Iterable[FraudLayerParameters] | None = None):
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel, threshold=conv_thresh)
        self.embed = nn.Linear(1, embed_dim)
        self.attn = ClassicalSelfAttention(embed_dim=embed_dim)
        # Default fraud parameters if none supplied
        if fraud_input_params is None:
            fraud_input_params = FraudLayerParameters(
                bs_theta=0.0, bs_phi=0.0, phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0), squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0), displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0))
        if fraud_layer_params is None:
            fraud_layer_params = []
        self.fraud = FraudDetectionModel(fraud_input_params, fraud_layer_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolution
        conv_feat = self.conv(x)                     # [batch, 1]
        # Linear embedding
        embedded = self.embed(conv_feat)             # [batch, embed_dim]
        # Self‑attention (batch dimension preserved)
        attn_out = self.attn(embedded.unsqueeze(1)).squeeze(1)  # [batch, embed_dim]
        # Reduce to 2‑D for fraud detection
        fraud_input = attn_out[:, :2]                # [batch, 2]
        return self.fraud(fraud_input)
