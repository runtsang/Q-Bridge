"""Hybrid classical-quantum binary classifier with fraud detection and self‑attention layers.

This module defines a PyTorch implementation that mirrors the structure of the
quantum counterpart while incorporating classical fraud‑detection style
parameters and a self‑attention block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Iterable

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

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
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
    linear = nn.Linear(2, 2, bias=True)
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 84):
        super().__init__()
        self.embed_dim = embed_dim
        self.q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q(x).unsqueeze(1)  # (batch,1,embed)
        k = self.k(x).unsqueeze(1)
        v = self.v(x).unsqueeze(1)
        scores = torch.softmax(torch.bmm(q, k.transpose(1, 2)) / np.sqrt(self.embed_dim), dim=-1)
        return torch.bmm(scores, v).squeeze(1)

class HybridFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    def __init__(self, in_features: int, shift: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

class HybridQuantumBinaryClassifier(nn.Module):
    """
    A hybrid classifier that merges a classical CNN, fraud‑detection style layers,
    a self‑attention block, and a differentiable sigmoid head.
    """

    def __init__(
        self,
        fraud_input: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)

        # Fraud‑detection module
        self.fraud_proj = nn.Linear(84, 2)
        self.fraud_module = build_fraud_detection_program(fraud_input, fraud_layers)

        # Self‑attention
        self.attention = ClassicalSelfAttention(embed_dim=84)

        # Final head
        self.fc3 = nn.Linear(84 + 1 + 84, 1)
        self.hybrid = Hybrid(1, shift=shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(inputs))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))

        # Fraud detection path
        proj = self.fraud_proj(x)
        fraud_out = self.fraud_module(proj).squeeze(-1)

        # Self‑attention path
        attn_out = self.attention(x)

        # Concatenate and head
        concat = torch.cat((x, fraud_out.unsqueeze(-1), attn_out), dim=-1)
        logits = self.fc3(concat)
        probs = self.hybrid(logits)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = [
    "FraudLayerParameters",
    "HybridQuantumBinaryClassifier",
    "HybridFunction",
    "Hybrid",
]
