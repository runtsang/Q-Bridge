"""Fraud detection model with supervised training and hybrid loss."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import numpy as np
import torch.nn.functional as F


class FraudDataset(Dataset):
    """Dataset that loads a simple tabular CSV with one class label – value is 0 or 1."""

    def __init__(self, csv_path: str) -> None:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        self.features = torch.tensor(data[:, :-1], dtype=torch.float32)
        self.labels = torch.tensor(data[:, -1], dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""

    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionHybridModel(nn.Module):
    """Hybrid model that fuses classical logits with quantum expectation values."""

    def __init__(self, ml_model: nn.Module, qml_module: nn.Module):
        super().__init__()
        self.ml = ml_model
        self.qml = qml_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.ml(x)
        q_expect = self.qml(x)
        return logits + q_expect


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 5,
) -> List[float]:
    """Training loop with early stopping based on validation loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        with torch.no_grad():
            epoch_val_loss = 0.0
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = F.binary_cross_entropy_with_logits(preds, batch_y)
                epoch_val_loss += loss.item() * batch_x.size(0)

            epoch_val_loss /= len(val_loader.dataset)
            val_losses.append(epoch_val_loss)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return train_losses, val_losses


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybridModel",
    "FraudDataset",
    "train",
]
