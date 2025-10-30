"""Enhanced classical fraud‑detection model with dropout, batch‑norm and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import nn, optim
from torch.nn import functional as F


@dataclass
class FraudLayerParameters:
    """Parameters for a single classical layer that mimics a photonic block."""
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


class FraudDetectionModel(nn.Module):
    """Hybrid‑style fraud‑detection model that can be trained end‑to‑end.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters of the first photonic‑like layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers.
    dropout : float, optional
        Dropout probability applied after each layer (default: 0.0).
    batch_norm : bool, optional
        Whether to insert a BatchNorm2d after each linear transformation.
    """
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        for layer in layers:
            modules.append(_layer_from_params(layer, clip=True))
            if batch_norm:
                modules.append(nn.BatchNorm1d(2))
            if dropout > 0.0:
                modules.append(nn.Dropout(dropout))
        modules.append(nn.Linear(2, 1))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @staticmethod
    def from_params(
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        *,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ) -> "FraudDetectionModel":
        """Convenience constructor that accepts raw parameter objects."""
        return FraudDetectionModel(
            input_params, layers, dropout=dropout, batch_norm=batch_norm
        )

    def train_one_epoch(
        self,
        dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device | str = "cpu",
    ) -> float:
        """Simple training loop for one epoch."""
        self.train()
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = self(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(dataloader.dataset)

    def evaluate(
        self,
        dataloader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        criterion: nn.Module,
        device: torch.device | str = "cpu",
    ) -> float:
        """Evaluation loop without gradient computation."""
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                pred = self(x).squeeze()
                loss = criterion(pred, y)
                total_loss += loss.item() * x.size(0)
        return total_loss / len(dataloader.dataset)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
