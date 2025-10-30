"""
FraudDetectionHybrid: Classical hybrid model combining an autoencoder and a
photonic‑inspired classifier.

This module pulls together the best ideas from the seed projects:
  * The dense encoder/decoder architecture from the Autoencoder seed.
  * The layer construction logic (weight clipping, custom activations) from
    the FraudDetection seed.
  * A clean public API that mirrors the structure of the quantum version
    to make the two implementations interchangeable.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

# Import the autoencoder components from the second reference pair.
# The path assumes the Autoencoder.py module lives in the same package.
from Autoencoder import AutoencoderNet, AutoencoderConfig


@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic‑inspired layer."""
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


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """
    Translate `FraudLayerParameters` into a PyTorch module that mimics the
    photonic circuit: a linear map, a tanh activation, and a final affine
    rescaling.  Clipping is applied to the weight matrix and bias vector
    when requested.
    """
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
            return outputs * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Build a sequential PyTorch model that emulates the layered photonic
    circuit.  The first layer receives the encoded latent vector; subsequent
    layers are clipped to keep the parameters within a physically realistic
    regime.
    """
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model.

    Parameters
    ----------
    autoencoder : AutoencoderNet
        Pre‑trained autoencoder that maps raw transaction features to a
        low‑dimensional latent space.
    input_params : FraudLayerParameters
        Parameters for the first photonic‑inspired layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for the remaining classifier layers.

    The forward pass first encodes the input with the autoencoder, then
    runs the classifier on the latent representation.
    """
    def __init__(
        self,
        autoencoder: AutoencoderNet,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
    ) -> None:
        super().__init__()
        self.autoencoder = autoencoder
        self.classifier = build_fraud_detection_program(input_params, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then classify."""
        latent = self.autoencoder.encode(x)
        return self.classifier(latent)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Expose the encoder for feature extraction."""
        return self.autoencoder.encode(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Expose the decoder for reconstruction."""
        return self.autoencoder.decode(z)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
]
