"""
Hybrid self‑attention module that combines quanvolution, fraud‑detection preprocessing,
and a classical multi‑head attention core.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 1. Quanvolution filter (from reference 3) ----
class QuanvolutionFilter(nn.Module):
    """Two‑pixel patch extraction followed by a 2‑D convolution."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (B, 1, H, W)  -> (B, 4, H/2, W/2)
        features = self.conv(x)
        return features.view(x.shape[0], -1)  # flatten patches


# ---- 2. Fraud‑detection style preprocessing (from reference 2) ----
class FraudLayerParameters:
    """Container for fraud‑detection layer parameters."""
    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: tuple[float, float],
        squeeze_r: tuple[float, float],
        squeeze_phi: tuple[float, float],
        displacement_r: tuple[float, float],
        displacement_phi: tuple[float, float],
        kerr: tuple[float, float],
    ):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr


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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a fraud‑detection style sequential module."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# ---- 3. Classical self‑attention helper (from reference 1) ----
class ClassicalSelfAttention:
    """Simple multi‑head attention block."""
    def __init__(self, embed_dim: int, num_heads: int = 1) -> None:
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, Seq, E)
        attn_output, _ = self.attn(x, x, x)
        return attn_output


# ---- 4. Hybrid SelfAttentionHybrid module ----
class SelfAttentionHybrid(nn.Module):
    """
    Combines quanvolution, fraud‑detection preprocessing, and classical self‑attention.
    """
    def __init__(
        self,
        embed_dim: int = 4,
        num_heads: int = 1,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        self.fraud = build_fraud_detection_program(
            FraudLayerParameters(
                bs_theta=0.5,
                bs_phi=0.5,
                phases=(0.1, 0.1),
                squeeze_r=(0.2, 0.2),
                squeeze_phi=(0.3, 0.3),
                displacement_r=(0.4, 0.4),
                displacement_phi=(0.5, 0.5),
                kerr=(0.6, 0.6),
            ),
            fraud_params or [],
        )
        self.attention = ClassicalSelfAttention(embed_dim, num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: (B, 1, H, W) grayscale images
        :return: (B, Seq, E) attention‑output
        """
        # 1. Quanvolution to extract patches
        patches = self.quanvolution(x)  # (B, N)
        # 2. Fraud‑detection preprocessing
        fraud_out = self.fraud(patches.unsqueeze(-1))  # (B, N, 1)
        # 3. Reshape to sequence of embeddings
        seq = fraud_out.squeeze(-1).unsqueeze(1)  # (B, 1, N)
        # 4. Self‑attention
        attn_out = self.attention(seq)  # (B, 1, N)
        return attn_out.squeeze(1)  # (B, N)
__all__ = ["SelfAttentionHybrid"]
