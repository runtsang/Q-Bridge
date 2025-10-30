"""Hybrid QCNN model with clipped parameter initialization.

This module defines a classical neural network that mirrors the depth and
convolution/pooling pattern of the original QCNN.  Each layer’s weights are
initialised from a user‑supplied :class:`QCNNLayerParams` data‑class, allowing
easy tuning and reproducibility.  Parameters are clipped to a tight bound
(±5.0) to emulate the numerical safeguards seen in the fraud‑detection
analogues, ensuring stable training while preserving the expressive power
of the network."""
from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import List


@dataclass
class QCNNLayerParams:
    """Parameters for a single QCNN layer."""
    conv_weights: torch.Tensor  # shape (out, in)
    conv_bias: torch.Tensor     # shape (out,)
    pool_weights: torch.Tensor  # shape (out, in)
    pool_bias: torch.Tensor     # shape (out,)


def _clip_tensor(tensor: torch.Tensor, bound: float) -> torch.Tensor:
    """Clip tensor values to a symmetric bound."""
    return tensor.clamp(-bound, bound)


def _linear_from_params(weights: torch.Tensor, bias: torch.Tensor, clip: bool) -> nn.Module:
    """Create a linear layer from explicit weight and bias tensors."""
    linear = nn.Linear(weights.shape[1], weights.shape[0])
    with torch.no_grad():
        linear.weight.copy_(weights)
        linear.bias.copy_(bias)
    if clip:
        linear.weight.data = _clip_tensor(linear.weight.data, 5.0)
        linear.bias.data = _clip_tensor(linear.bias.data, 5.0)
    return linear


class QCNNHybridModel(nn.Module):
    """A classical neural network that mimics the QCNN structure with clipped parameters."""
    def __init__(self, layers: List[QCNNLayerParams]) -> None:
        super().__init__()
        modules: List[nn.Module] = []

        for layer in layers:
            # Convolution part
            modules.append(_linear_from_params(layer.conv_weights, layer.conv_bias, clip=False))
            modules.append(nn.Tanh())
            # Pooling part
            modules.append(_linear_from_params(layer.pool_weights, layer.pool_bias, clip=True))
            modules.append(nn.Tanh())

        modules.append(nn.Linear(layers[-1].pool_weights.shape[0], 1))
        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sigmoid(self.network(x))


def QCNNHybrid(layers: List[QCNNLayerParams]) -> QCNNHybridModel:
    """Factory that creates a QCNNHybridModel from a list of layer parameters."""
    return QCNNHybridModel(layers)


__all__ = ["QCNNHybrid", "QCNNHybridModel", "QCNNLayerParams"]
