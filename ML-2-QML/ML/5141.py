"""Hybrid classical‑quantum binary classifier that fuses convolution,
self‑attention, quantum kernel, and fraud‑detection inspired layers."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import quantum helpers
from QuantumHybridBinaryClassifier import HybridBinaryClassifier as QuantumHybridBinaryClassifier
from QuantumHybridBinaryClassifier import QuantumSelfAttention

# Classical self‑attention (adapted from reference 4)
class ClassicalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Simple dot‑product attention using random projections
        query = inputs @ torch.randn(self.embed_dim, self.embed_dim, device=inputs.device)
        key   = inputs @ torch.randn(self.embed_dim, self.embed_dim, device=inputs.device)
        value = inputs
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

# Fraud‑detection layer utilities (reference 2)
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
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
    bias  = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = weight.clamp(-5.0, 5.0)
        bias  = bias.clamp(-5.0, 5.0)
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

# Hyper‑parameters for fraud detection layers
_input_params = FraudLayerParameters(
    bs_theta=0.4, bs_phi=0.6,
    phases=(0.1, -0.2),
    squeeze_r=(0.3, 0.5),
    squeeze_phi=(0.4, -0.5),
    displacement_r=(0.7, 0.9),
    displacement_phi=(0.8, -1.0),
    kerr=(0.2, 0.3)
)
_layer_params = [
    FraudLayerParameters(
        bs_theta=0.2, bs_phi=0.3,
        phases=(0.05, -0.05),
        squeeze_r=(0.1, 0.2),
        squeeze_phi=(0.2, -0.2),
        displacement_r=(0.3, 0.4),
        displacement_phi=(0.4, -0.4),
        kerr=(0.1, 0.1)
    ),
    FraudLayerParameters(
        bs_theta=0.1, bs_phi=0.2,
        phases=(0.02, -0.02),
        squeeze_r=(0.05, 0.1),
        squeeze_phi=(0.1, -0.1),
        displacement_r=(0.15, 0.25),
        displacement_phi=(0.25, -0.25),
        kerr=(0.05, 0.05)
    ),
]

class HybridBinaryClassifier(nn.Module):
    """Hybrid classical‑quantum binary classifier combining convolution,
    self‑attention, quantum kernel, and fraud‑detection inspired layers."""
    def __init__(self) -> None:
        super().__init__()
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)

        # Embedding and attention
        self.embed_dim = 4
        self.embedding = nn.Linear(55815, self.embed_dim)
        self.classical_attention = ClassicalSelfAttention(self.embed_dim)
        self.quantum_attention = QuantumSelfAttention(n_qubits=self.embed_dim)
        # Random parameters for quantum attention
        self.rotation_params = np.random.randn(self.embed_dim * 3)
        self.entangle_params = np.random.randn(self.embed_dim - 1)

        # Linear mapping to fraud‑detection input (2‑dim)
        self.to_fraud = nn.Linear(self.embed_dim, 2)

        # Fraud‑detection inspired sequential model
        self.fraud_network = build_fraud_detection_program(_input_params, _layer_params)

        # Final linear layer before quantum head
        self.final_linear = nn.Linear(1, 1)

        # Quantum expectation head
        self.quantum_head = QuantumHybridBinaryClassifier(shift=np.pi/2, shots=200)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convolutional backbone
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)  # (batch, 55815)

        # Embedding
        x = self.embedding(x)  # (batch, embed_dim)

        # Classical self‑attention
        x = self.classical_attention(x)

        # Quantum self‑attention
        q_attn_output = []
        for sample in x.detach().cpu().numpy():
            q_output = self.quantum_attention.run(
                self.rotation_params,
                self.entangle_params,
                sample
            )
            q_attn_output.append(q_output)
        q_attn_output = torch.tensor(q_attn_output, dtype=x.dtype, device=x.device)
        x = x + q_attn_output

        # Map to fraud‑detection input
        x = self.to_fraud(x)  # (batch, 2)

        # Fraud‑detection network
        x = self.fraud_network(x)  # (batch, 1)

        # Final linear (identity)
        x = self.final_linear(x)  # (batch, 1)

        # Quantum expectation head
        prob = self.quantum_head(x)  # (batch, 1)

        # Binary probabilities
        return torch.cat((prob, 1 - prob), dim=-1)
