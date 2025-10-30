"""
FraudDetectionModel – Classical implementation with configurable depth, dropout and optional batch‑norm.

The model mirrors the photonic architecture but is fully trainable with PyTorch.
It is intentionally lightweight yet extensible: new layers can be appended, dropout
probability tuned, and batch‑norm enabled without changing the public API.

Key features
------------
* ``FraudLayerParameters`` – holds all parameters for a single layer.
* ``FraudDetectionModel`` – PyTorch ``nn.Module`` that builds a sequential network
  from an input layer followed by any number of hidden layers.
* Optional dropout and batch‑norm provide regularisation and normalisation.
* ``from_config`` helper allows construction from a dictionary or JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """
    Parameters describing a fully connected layer in the classical model.

    Attributes
    ----------
    bs_theta : float
        Weight for the first input feature.
    bs_phi : float
        Weight for the second input feature.
    phases : tuple[float, float]
        Bias values for the two output units.
    squeeze_r : tuple[float, float]
        Scaling factors applied after the linear transformation.
    squeeze_phi : tuple[float, float]
        Shift values applied after scaling.
    displacement_r : tuple[float, float]
        Additional scaling applied before the final linear layer.
    displacement_phi : tuple[float, float]
        Final shift applied before the output layer.
    kerr : tuple[float, float]
        Extra bias terms used for the output layer.
    """

    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """
    Convert a ``FraudLayerParameters`` instance into a PyTorch ``nn.Module``.

    Parameters
    ----------
    params : FraudLayerParameters
        Layer parameters.
    clip : bool
        Whether to clip weight and bias values to a safe range.
    """
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi], [params.squeeze_r[0], params.squeeze_r[1]]],
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

        def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inp))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """
    Create a sequential PyTorch model mirroring the layered structure.

    The first layer is built without clipping; subsequent layers are clipped to
    avoid exploding gradients.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the initial layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for the hidden layers.

    Returns
    -------
    nn.Sequential
        A fully constructed network ready for training.
    """
    modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionModel(nn.Module):
    """
    Configurable fraud‑detection network.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer.
    layers : Sequence[FraudLayerParameters]
        Parameters for subsequent hidden layers.
    dropout : float, optional
        Dropout probability applied after each hidden layer. 0.0 disables dropout.
    use_batchnorm : bool, optional
        If True, a 1‑D batch‑norm layer is added after each hidden layer.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Sequence[FraudLayerParameters],
        *,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        self._sequence = nn.ModuleList()
        # Input layer
        self._sequence.append(_layer_from_params(input_params, clip=False))

        # Hidden layers
        for layer_params in layers:
            layer = _layer_from_params(layer_params, clip=True)
            self._sequence.append(layer)
            if self.use_batchnorm:
                self._sequence.append(nn.BatchNorm1d(2))
            if self.dropout > 0.0:
                self._sequence.append(nn.Dropout(self.dropout))

        # Output layer
        self._sequence.append(nn.Linear(2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for module in self._sequence:
            x = module(x)
        return x

    @staticmethod
    def from_config(
        config: dict,
        *,
        dropout: float = 0.0,
        use_batchnorm: bool = False,
    ) -> "FraudDetectionModel":
        """
        Construct a model from a configuration dictionary.

        The dictionary should contain ``input_params`` and ``layers`` keys,
        each mapping to a list of parameter dictionaries matching
        ``FraudLayerParameters``.
        """
        input_params = FraudLayerParameters(**config["input_params"])
        layers = [FraudLayerParameters(**p) for p in config["layers"]]
        return FraudDetectionModel(
            input_params, layers, dropout=dropout, use_batchnorm=use_batchnorm
        )

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionModel"]
