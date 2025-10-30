from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


@dataclass
class AttentionParameters:
    """Parameters for a self‑attention block."""
    rotation_params: np.ndarray
    entangle_params: np.ndarray


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


class ClassicalSelfAttention(nn.Module):
    """A lightweight self‑attention block that mirrors the quantum interface."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        query = torch.matmul(
            inputs,
            torch.as_tensor(rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32),
        )
        key = torch.matmul(
            inputs,
            torch.as_tensor(entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32),
        )
        value = inputs
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value


class AttentionWrapper(nn.Module):
    """Wraps the classical self‑attention to be used inside nn.Sequential."""
    def __init__(self, attention: ClassicalSelfAttention, params: AttentionParameters):
        super().__init__()
        self.attention = attention
        self.params = params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(
            self.params.rotation_params,
            self.params.entangle_params,
            x,
        )


class FraudDetectionML(nn.Module):
    """Hybrid fraud‑detection model that combines classical self‑attention with
    photonic‑inspired layers. The attention block is applied first, and its
    output is fed into a sequence of custom linear transforms."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        attention_params: AttentionParameters,
    ) -> None:
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim=4)
        modules: list[nn.Module] = []
        modules.append(AttentionWrapper(self.attention, attention_params))
        modules.append(_layer_from_params(input_params, clip=False))
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_fraud_detection_ml(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    attention_params: AttentionParameters,
) -> FraudDetectionML:
    """Convenience constructor that returns a fully‑wired FraudDetectionML."""
    return FraudDetectionML(input_params, layers, attention_params)
