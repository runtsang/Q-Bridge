"""Hybrid fraud detection model – classical component.

This module implements a deep neural network that mirrors the photonic
layer structure while adding modern regularisation (dropout, batch
normalisation) and a probabilistic sampler.  The resulting class,
`FraudDetectionHybrid`, exposes a forward pass that can be used for
training or inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# 1. Parameter container – identical to the photonic definition
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters for a single fully‑connected layer inspired by photonic gates."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clips a real value to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


# --------------------------------------------------------------------------- #
# 2. Classical layer builder – extends the seed with regularisation
# --------------------------------------------------------------------------- #
def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Construct a single classical layer that mimics a photonic block."""
    # Linear transformation derived from beam‑splitter and squeezing angles
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

    # Regularised activation
    activation = nn.ReLU()
    # Scale/shift to emulate displacement
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    # Wrap everything in a sub‑module
    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.dropout = nn.Dropout(p=0.1)
            self.batch_norm = nn.BatchNorm1d(2)
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.linear(inputs)
            x = self.activation(x)
            x = self.batch_norm(x)
            x = self.dropout(x)
            x = x * self.scale + self.shift
            return x

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Build a PyTorch sequential model that follows the layered
    architecture of the photonic circuit.  The first layer is left unclipped
    to preserve raw signal, subsequent layers are clipped to avoid exploding
    weights.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# 3. Probabilistic sampler – classical analogue of the Qiskit SamplerQNN
# --------------------------------------------------------------------------- #
class SamplerQNN(nn.Module):
    """Softmax classifier mimicking a quantum sampler."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)


# --------------------------------------------------------------------------- #
# 4. Fully‑connected quantum layer – classical stand‑in
# --------------------------------------------------------------------------- #
class FCL(nn.Module):
    """Single‑parameter linear mapping used as a placeholder for a quantum layer."""

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> float:
        theta_tensor = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(theta_tensor)).mean()
        return expectation.item()


# --------------------------------------------------------------------------- #
# 5. Hybrid fraud‑detection model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """
    End‑to‑end classical model that

    1. extracts features with a photonic‑inspired network,
    2. maps the features to class probabilities via a sampler,
    3. applies a final sigmoid for binary output.

    The architecture can be trained end‑to‑end using standard PyTorch optimisers.
    """

    def __init__(self, input_params: FraudLayerParameters, hidden_params: Sequence[FraudLayerParameters]) -> None:
        super().__init__()
        self.feature_extractor = build_fraud_detection_program(input_params, hidden_params)
        self.classifier = SamplerQNN()
        self.final = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.feature_extractor(x)
        probs = self.classifier(features)
        logits = self.final(probs)
        return torch.sigmoid(logits)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper that returns the fraud probability."""
        return self.forward(x).squeeze(-1)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "SamplerQNN", "FCL", "FraudDetectionHybrid"]
