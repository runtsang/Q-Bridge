"""Classical neural network for fraud detection with extended capabilities.

The model mirrors the photonic architecture but adds dropout, L2 regularisation,
dynamic learning‑rate scheduling and a convenience training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
import math


@dataclass
class LayerParams:
    """Hyper‑parameters of a single fully‑connected layer."""
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


class _ScaleShift(nn.Module):
    """Applies element‑wise scaling and shifting."""
    def __init__(self, scale: torch.Tensor, shift: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.shift


def _layer_from_params(
    params: LayerParams,
    *,
    clip: bool,
    dropout: float,
) -> nn.Module:
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

    linear = nn.Linear(2, 2, bias=True)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    return nn.Sequential(
        linear,
        nn.Tanh(),
        nn.Dropout(dropout),
        _ScaleShift(scale, shift),
    )


class FraudDetectionHybrid(nn.Module):
    """Hybrid‑style neural network for fraud detection.

    Parameters
    ----------
    input_params : LayerParams
        Parameters of the first layer.
    layers : Iterable[LayerParams]
        Parameters of the hidden layers.
    dropout_rate : float, optional
        Dropout probability applied after each hidden layer.
    l2_lambda : float, optional
        L2 weight‑decay regularisation strength.
    """

    def __init__(
        self,
        input_params: LayerParams,
        layers: Iterable[LayerParams],
        *,
        dropout_rate: float = 0.2,
        l2_lambda: float = 0.01,
    ) -> None:
        super().__init__()
        modules: list[nn.Module] = [
            _layer_from_params(input_params, clip=False, dropout=dropout_rate)
        ]
        modules.extend(
            _layer_from_params(l, clip=True, dropout=dropout_rate) for l in layers
        )
        modules.append(nn.Linear(2, 1))
        modules.append(nn.Sigmoid())
        self.network = nn.Sequential(*modules)
        self.l2_lambda = l2_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def train_one_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module = nn.BCELoss(),
        optimizer: optim.Optimizer | None = None,
        scheduler: optim.lr_scheduler._LRScheduler | None = None,
        device: torch.device | str = "cpu",
    ) -> tuple[float, float]:
        """Train for one epoch and return average loss / accuracy."""
        self.train()
        if optimizer is None:
            optimizer = optim.Adam(
                self.parameters(), lr=1e-3, weight_decay=self.l2_lambda
            )
        if scheduler is None:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        total_loss = 0.0
        correct = 0
        n_samples = 0
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = self(X).squeeze()
            loss = criterion(preds, y.float())
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * X.size(0)
            preds_bin = (preds > 0.5).float()
            correct += (preds_bin == y).sum().item()
            n_samples += X.size(0)

        avg_loss = total_loss / n_samples
        accuracy = correct / n_samples
        return avg_loss, accuracy

    def evaluate(
        self,
        dataloader: DataLoader,
        device: torch.device | str = "cpu",
    ) -> tuple[float, float]:
        """Return average loss and accuracy on validation data."""
        self.eval()
        total_loss = 0.0
        correct = 0
        n_samples = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                preds = self(X).squeeze()
                loss = F.binary_cross_entropy(preds, y.float())
                total_loss += loss.item() * X.size(0)
                preds_bin = (preds > 0.5).float()
                correct += (preds_bin == y).sum().item()
                n_samples += X.size(0)
        avg_loss = total_loss / n_samples
        accuracy = correct / n_samples
        return avg_loss, accuracy


__all__ = ["LayerParams", "FraudDetectionHybrid"]
