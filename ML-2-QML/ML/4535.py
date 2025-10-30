"""Unified hybrid classifier – classical counterpart.

The design merges three proven concepts:
* 1️⃣ A deep residual MLP that accepts the raw feature vector and produces a 2‑dimensional embedding.
* 2️⃣ A photonic‑style layer that applies a learnable linear transformation followed by a Tanh, a scaling/shift, and a final linear head – mimicking the fraud‑detection photonic circuit.
* 3️⃣ A lightweight sampler‑style network that outputs class probabilities via softmax.

All three components are chained so that the residual MLP feeds the photonic layer, which in turn feeds the sampler head.  The module is fully PyTorch‑compatible and can be trained with standard optimisers.

The public API mirrors the quantum helper: ``build_hybrid_classifier`` returns the network, a list of parameter names for the encoder, and the observable indices (here simply ``[0, 1]``).  The helper also exposes ``get_parameter_dict`` for introspection and ``freeze_encoder`` to freeze the classical encoder during quantum‑only experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class _LayerParams:
    bs_theta: float = 0.0
    bs_phi: float = 0.0
    phases: Tuple[float, float] = (0.0, 0.0)
    squeeze_r: Tuple[float, float] = (0.0, 0.0)
    squeeze_phi: Tuple[float, float] = (0.0, 0.0)
    displacement_r: Tuple[float, float] = (0.0, 0.0)
    displacement_phi: Tuple[float, float] = (0.0, 0.0)
    kerr: Tuple[float, float] = (0.0, 0.0)


def _layer_from_params(params: _LayerParams, clip: bool) -> nn.Module:
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


class _ResidualMLP(nn.Module):
    """Simple residual block used in the encoder."""

    def __init__(self, in_features: int, hidden: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden)
        self.linear2 = nn.Linear(hidden, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.linear1(x))
        out = self.linear2(out)
        return F.relu(out + residual)


class UnifiedHybridClassifier(nn.Module):
    """Full hybrid classifier with classical encoder, photonic layer, and sampler head."""

    def __init__(
        self,
        in_features: int = 2,
        hidden: int = 8,
        depth: int = 2,
        fraud_params: _LayerParams = _LayerParams(),
        sampler_hidden: int = 4,
    ) -> None:
        super().__init__()
        # Encoder
        blocks = [_ResidualMLP(in_features, hidden) for _ in range(depth)]
        self.encoder = nn.Sequential(*blocks)
        # Photonic‑style fraud layer
        self.fraud_layer = _layer_from_params(fraud_params, clip=False)
        # Sampler head
        self.sampler = nn.Sequential(
            nn.Linear(2, sampler_hidden),
            nn.Tanh(),
            nn.Linear(sampler_hidden, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.fraud_layer(x)
        logits = self.sampler(x)
        return F.softmax(logits, dim=-1)

    def get_parameter_dict(self) -> dict:
        """Return a mapping from parameter names to tensors."""
        return {name: param for name, param in self.named_parameters()}

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters (useful for quantum‑only experiments)."""
        for param in self.encoder.parameters():
            param.requires_grad = False


__all__ = ["UnifiedHybridClassifier"]
