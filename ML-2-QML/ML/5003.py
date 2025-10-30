"""Hybrid classical module combining CNN, self‑attention, fraud‑detection style layers, and a sigmoid head."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

# ----------------------------------------------------------------------
# Fraud‑detection style linear layer
# ----------------------------------------------------------------------
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
        dtype=torch.float32)
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

def build_fraud_detection_module(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch module mirroring the fraud‑detection style layers."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# Self‑attention helper
# ----------------------------------------------------------------------
def SelfAttention(embed_dim: int = 4):
    class ClassicalSelfAttention:
        def __init__(self, embed_dim: int) -> None:
            self.embed_dim = embed_dim

        def run(
            self,
            rotation_params: np.ndarray,
            entangle_params: np.ndarray,
            inputs: np.ndarray,
        ) -> np.ndarray:
            query = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            key   = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32)
            value = torch.as_tensor(inputs, dtype=torch.float32)
            scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
            return (scores @ value).numpy()
    return ClassicalSelfAttention(embed_dim)

# ----------------------------------------------------------------------
# Hybrid sigmoid head
# ----------------------------------------------------------------------
class HybridFunction(torch.autograd.Function):
    """Differentiable sigmoid activation that mimics a quantum expectation head."""
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, shift: float) -> torch.Tensor:  # type: ignore[override]
        outputs = torch.sigmoid(inputs + shift)
        ctx.save_for_backward(outputs)
        return outputs

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (outputs,) = ctx.saved_tensors
        grad_inputs = grad_output * outputs * (1 - outputs)
        return grad_inputs, None

class Hybrid(nn.Module):
    """Dense head that replaces a quantum circuit in the original model."""
    def __init__(self, in_features: int, shift: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.shift = shift

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        logits = inputs.view(inputs.size(0), -1)
        return HybridFunction.apply(self.linear(logits), self.shift)

# ----------------------------------------------------------------------
# Main hybrid model
# ----------------------------------------------------------------------
class QuantumNATEnhanced(nn.Module):
    """CNN backbone + fraud‑detection style head + self‑attention + hybrid sigmoid."""
    def __init__(self, fraud_params: FraudLayerParameters, fraud_layers: Iterable[FraudLayerParameters]) -> None:
        super().__init__()
        # Convolutional backbone
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Self‑attention
        self.attention = SelfAttention(embed_dim=4)
        # Fraud‑detection style module
        self.fraud_head = build_fraud_detection_module(fraud_params, fraud_layers)
        # Fully connected head
        self.fc = nn.Sequential(nn.Linear(16 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 4))
        self.norm = nn.BatchNorm1d(4)
        # Hybrid sigmoid
        self.hybrid = Hybrid(in_features=4, shift=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # Feature extraction
        features = self.features(x)
        flat = features.view(features.size(0), -1)
        # Self‑attention on raw features
        attn_out = self.attention.run(
            rotation_params=np.random.rand(4 * 4),
            entangle_params=np.random.rand(4 * 4),
            inputs=flat.cpu().numpy()
        )
        attn_tensor = torch.from_numpy(attn_out).to(x.device)
        # Fraud‑detection style processing
        fraud_out = self.fraud_head(attn_tensor)
        # FC head
        fc_out = self.fc(fraud_out)
        normed = self.norm(fc_out)
        # Hybrid sigmoid to produce probabilities
        probs = self.hybrid(normed)
        return torch.cat((probs, 1 - probs), dim=-1)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_module",
    "SelfAttention",
    "HybridFunction",
    "Hybrid",
    "QuantumNATEnhanced",
]
