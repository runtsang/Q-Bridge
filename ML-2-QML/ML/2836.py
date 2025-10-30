"""Hybrid classical self‑attention and fraud detection framework."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

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

class SelfAttention:
    """Hybrid classical self‑attention and fraud detection model."""
    def __init__(
        self,
        embed_dim: int = 4,
        n_qubits: int = 4,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layer_params: Iterable[FraudLayerParameters] | None = None,
    ):
        self.embed_dim = embed_dim
        self.n_qubits = n_qubits
        self.attention = self._build_classical_attention()
        self.fraud_model = None
        if fraud_input_params is not None and fraud_layer_params is not None:
            self.fraud_model = build_fraud_detection_program(fraud_input_params, fraud_layer_params)

    def _build_classical_attention(self) -> nn.Module:
        class ClassicalSelfAttention(nn.Module):
            def __init__(self, embed_dim: int):
                super().__init__()
                self.embed_dim = embed_dim

            def forward(
                self,
                rotation_params: np.ndarray,
                entangle_params: np.ndarray,
                inputs: torch.Tensor,
            ) -> torch.Tensor:
                query = torch.as_tensor(
                    inputs @ rotation_params.reshape(self.embed_dim, -1),
                    dtype=torch.float32,
                )
                key = torch.as_tensor(
                    inputs @ entangle_params.reshape(self.embed_dim, -1),
                    dtype=torch.float32,
                )
                value = inputs
                scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
                return scores @ value

        return ClassicalSelfAttention(self.embed_dim)

    def run_classical_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        tensor_inputs = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.attention(rotation_params, entangle_params, tensor_inputs)
        return out.detach().cpu().numpy()

    def run_classical_fraud(self, inputs: np.ndarray) -> np.ndarray:
        if self.fraud_model is None:
            raise RuntimeError("Fraud model not configured.")
        tensor_inputs = torch.as_tensor(inputs, dtype=torch.float32)
        out = self.fraud_model(tensor_inputs)
        return out.detach().cpu().numpy()

    def __repr__(self) -> str:
        return f"<SelfAttention embed_dim={self.embed_dim} n_qubits={self.n_qubits}>"

__all__ = ["SelfAttention", "FraudLayerParameters", "build_fraud_detection_program"]
