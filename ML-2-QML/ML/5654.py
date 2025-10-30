"""Extended fraud‑detection pipeline combining feature engineering and a dense neural network.

The implementation expands the original seed by adding:
* A simple `FeatureEncoder` that scales and centres the raw inputs.
* A configurable `FraudClassifier` that can switch between sigmoid and softmax heads.
* Dropout to regularise the model during training.
* Convenience methods for one‑step training and evaluation.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionExtended",
    "train_step",
    "evaluate",
]

# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
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


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


# --------------------------------------------------------------------------- #
# Layer construction helpers
# --------------------------------------------------------------------------- #
def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single linear layer whose weights and bias are derived from
    the photonic parameters.  The original seed used a Tanh activation; we keep
    that behaviour but expose a drop‑out option in the higher‑level model.
    """
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


# --------------------------------------------------------------------------- #
# Main model
# --------------------------------------------------------------------------- #
class FraudDetectionExtended(nn.Module):
    """Hybrid classical fraud‑detection model that mirrors the photonic circuit
    but adds a learnable encoder, dropout and an adjustable output head.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        num_features: int,
        dropout: float = 0.3,
        output_activation: str = "sigmoid",
    ) -> None:
        super().__init__()
        # Feature encoder: map raw transaction data to the 2‑dimensional space
        # the photonic layers expect.  A simple linear mapping with
        # batch‑normalisation suffices for illustration.
        self.encoder = nn.Sequential(
            nn.Linear(num_features, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(),
        )

        # Build the sequence of photonic‑derived layers
        self.layers = nn.ModuleList(
            [_layer_from_params(input_params, clip=False)]
            + [_layer_from_params(lp, clip=True) for lp in layers]
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2, 1)
        if output_activation == "sigmoid":
            self.out_act = nn.Sigmoid()
        elif output_activation == "softmax":
            self.out_act = nn.Softmax(dim=1)
        else:
            raise ValueError(f"Unsupported activation {output_activation!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return self.out_act(x)

# --------------------------------------------------------------------------- #
# Convenience helpers
# --------------------------------------------------------------------------- #
def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    data: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Perform a single optimisation step and return the loss value."""
    optimizer.zero_grad()
    preds = model(data)
    loss = loss_fn(preds.squeeze(), labels.float())
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model: nn.Module, data: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the binary classification accuracy on the given batch."""
    with torch.no_grad():
        preds = model(data)
        preds = (preds.squeeze() > 0.5).float()
        return (preds == labels).float().mean().item()
