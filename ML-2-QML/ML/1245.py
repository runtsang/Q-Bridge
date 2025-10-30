"""Classical fraud detection model using PyTorch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    trainable: bool = False


class _ScaleShift(nn.Module):
    """Scale‑and‑shift transformation applied after the activation."""
    def __init__(self, scale: torch.Tensor, shift: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


class FraudDetectionModel(nn.Module):
    """Neural network that mirrors the photonic fraud‑detection architecture.

    Parameters
    ----------
    layer_params
        Iterable of :class:`FraudLayerParameters` describing each layer.
    dropout_rate
        Dropout probability applied after every activation.
    l2_reg
        L2 regularisation coefficient applied to all trainable weights.
    """
    def __init__(
        self,
        layer_params: Iterable[FraudLayerParameters],
        dropout_rate: float = 0.0,
        l2_reg: float = 0.0,
    ) -> None:
        super().__init__()
        self.l2_reg = l2_reg
        modules: List[nn.Module] = []

        for params in layer_params:
            weight = torch.tensor(
                [
                    [params.bs_theta, params.bs_phi],
                    [params.squeeze_r[0], params.squeeze_r[1]],
                ],
                dtype=torch.float32,
            )
            bias = torch.tensor(params.phases, dtype=torch.float32)

            linear = nn.Linear(2, 2, bias=True)
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)

            linear.weight.requires_grad = params.trainable
            linear.bias.requires_grad = params.trainable

            modules.append(linear)
            modules.append(nn.Tanh())
            if dropout_rate > 0.0:
                modules.append(nn.Dropout(dropout_rate))

            scale = torch.tensor(params.displacement_r, dtype=torch.float32)
            shift = torch.tensor(params.displacement_phi, dtype=torch.float32)
            modules.append(_ScaleShift(scale, shift))

        modules.append(nn.Linear(2, 1))
        self.network = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def l2_loss(self) -> torch.Tensor:
        """Return L2 penalty over all trainable weights."""
        if self.l2_reg == 0.0:
            return torch.tensor(0.0, device=self.network[0].weight.device)
        l2 = torch.tensor(0.0, device=self.network[0].weight.device)
        for m in self.network:
            if isinstance(m, nn.Linear) and m.weight.requires_grad:
                l2 += torch.norm(m.weight, p=2) ** 2
        return self.l2_reg * l2


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
