"""Hybrid fraud‑detection model – classical branch with advanced regularisation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters of a single fully‑connected layer."""
    in_features: int
    out_features: int
    bias: bool = True
    dropout_prob: float = 0.0
    use_batchnorm: bool = False
    weight_decay: float = 0.0


def _make_layer(params: FraudLayerParameters) -> nn.Module:
    layers: list[nn.Module] = []

    linear = nn.Linear(params.in_features, params.out_features, bias=params.bias)
    nn.init.kaiming_uniform_(linear.weight, a=math.sqrt(5))
    if params.bias:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(linear.bias, -bound, bound)

    layers.append(linear)

    if params.use_batchnorm:
        layers.append(nn.BatchNorm1d(params.out_features))

    layers.append(nn.ReLU(inplace=True))

    if params.dropout_prob > 0.0:
        layers.append(nn.Dropout(p=params.dropout_prob))

    return nn.Sequential(*layers)


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a sequential model mirroring the layered structure with optional regularisation."""
    modules = [_make_layer(input_params)]
    modules.extend(_make_layer(layer) for layer in layers)
    modules.append(nn.Linear(layers[-1].out_features, 1))
    return nn.Sequential(*modules)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program"]
