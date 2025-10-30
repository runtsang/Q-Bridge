import torch
from torch import nn
import numpy as np
from dataclasses import dataclass

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
    clip: bool = False

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if params.clip:
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
        def __init__(self):
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

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block compatible with the quantum interface."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
    ) -> torch.Tensor:
        query = inputs @ rotation_params.reshape(self.embed_dim, -1)
        key = inputs @ entangle_params.reshape(self.embed_dim, -1)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return scores @ inputs

class EstimatorNN(nn.Module):
    """Small feed‑forward regressor used in the hybrid pipeline."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model that merges:
      * photonic‑style classical layers (from FraudLayerParameters)
      * a classical self‑attention block
      * a lightweight feed‑forward network
      * optional quantum feature vectors supplied at inference time
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: list[FraudLayerParameters],
        attention: ClassicalSelfAttention | None = None,
        quantum_feature_dim: int = 0,
    ):
        super().__init__()
        modules = [_layer_from_params(input_params)]
        modules.extend(_layer_from_params(layer) for layer in layers)
        if attention is None:
            attention = ClassicalSelfAttention(embed_dim=4)
        self.attention = attention
        self.classical_body = nn.Sequential(*modules, nn.Linear(2, 1))
        total_input_dim = 1 + attention.embed_dim + quantum_feature_dim
        self.feedforward = EstimatorNN(total_input_dim)

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor,
        entangle_params: torch.Tensor,
        quantum_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # classical photonic layers
        x = self.classical_body(inputs)
        # self‑attention
        attn = self.attention(inputs, rotation_params, entangle_params)
        x = torch.cat([x, attn], dim=-1)
        # optional quantum features
        if quantum_features is not None:
            x = torch.cat([x, quantum_features], dim=-1)
        return self.feedforward(x)

__all__ = [
    "FraudLayerParameters",
    "ClassicalSelfAttention",
    "EstimatorNN",
    "FraudDetectionHybrid",
]
