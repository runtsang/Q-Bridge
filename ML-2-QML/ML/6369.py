"""
Enhanced classical fraud detection model.

This module implements the FraudDetectionEnhanced class as a PyTorch neural network
that extends the original two‑layer linear stack with a learnable feature extractor
and multi‑class support.  The architecture is fully differentiable and can be
trained end‑to‑end with standard optimizers.

The class exposes a minimal interface:
  * forward(inputs) -> logits
  * predict(inputs) -> class indices
  * loss(inputs, targets) -> loss tensor
  * parameters() -> iterator over trainable parameters
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, clip: bool = True) -> nn.Module:
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

class FraudDetectionEnhanced(nn.Module):
    """Two‑stage classical fraud detection network.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the feature‑extractor layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for the subsequent stacked layers.
    num_classes : int, default=2
        Number of output classes.  If ``num_classes`` is 1 the network outputs a
        single logit suitable for binary classification.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        # Feature extractor
        self.feature_extractor = _layer_from_params(input_params, clip=False)
        # Stacked layers
        self.stacked = nn.Sequential(
            *(_layer_from_params(l, clip=True) for l in layers)
        )
        # Final classification head
        self.num_classes = num_classes
        if num_classes == 1:
            self.head = nn.Linear(2, 1)
        else:
            self.head = nn.Linear(2, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        x = self.feature_extractor(inputs)
        x = self.stacked(x)
        logits = self.head(x)
        return logits

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return class indices."""
        logits = self.forward(inputs)
        if self.num_classes == 1:
            probs = torch.sigmoid(logits.squeeze(-1))
            return (probs > 0.5).long()
        return torch.argmax(logits, dim=-1)

    def loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return cross‑entropy or binary cross‑entropy loss."""
        logits = self.forward(inputs)
        if self.num_classes == 1:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), targets.float()
            )
        else:
            loss = nn.functional.cross_entropy(logits, targets)
        return loss

    def parameters(self):
        """Iterator over trainable parameters."""
        return super().parameters()

__all__ = ["FraudLayerParameters", "FraudDetectionEnhanced"]
