"""Enhanced classical fraud detection model with dropout, batchnorm and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters for a fully‑connected layer plus regularisation knobs."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    dropout: float = 0.0
    batchnorm: bool = False


def _clip(value: torch.Tensor, bound: float) -> torch.Tensor:
    return torch.clamp(value, -bound, bound)


def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [
            [params.bs_theta, params.bs_phi],
            [params.squeeze_r[0], params.squeeze_r[1]],
        ],
        dtype=torch.float32,
    )
    bias = torch.tensor(params.phases, dtype=torch.float32)
    if clip:
        weight = _clip(weight, 5.0)
        bias = _clip(bias, 5.0)

    linear = nn.Linear(2, 2)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)

    modules: list[nn.Module] = [linear, nn.Tanh()]
    if params.batchnorm:
        modules.append(nn.BatchNorm1d(2))
    if params.dropout > 0.0:
        modules.append(nn.Dropout(params.dropout))

    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seq = nn.Sequential(*modules)
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.seq(inputs)
            return out * self.scale + self.shift

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Build a sequential model mirroring the photonic architecture."""
    seq: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
    seq += [_layer_from_params(l, clip=True) for l in layers]
    seq.append(nn.Linear(2, 1))
    return nn.Sequential(*seq)


def train_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device = torch.device("cpu"),
) -> None:
    """Simple binary cross‑entropy training loop."""
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()


def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Return accuracy over a data loader."""
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device).float().unsqueeze(1)
            logits = model(x)
            predictions = torch.sigmoid(logits) > 0.5
            correct += predictions.eq(y).sum().item()
            total += y.numel()
    return correct / total


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "train_model",
    "evaluate_model",
]
