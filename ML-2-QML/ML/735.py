"""Augmented classical fraud detection model with training utilities and dropout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn, optim


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
    dropout: float = 0.0


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
            self.dropout = nn.Dropout(params.dropout)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return self.dropout(outputs)

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


class FraudDetector(nn.Module):
    """End‑to‑end fraud detection model with optional dropout and training helpers."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base = build_fraud_detection_program(input_params, layers)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        self.classifier = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.dropout(self.base(x))
        logits = self.classifier(x)
        return torch.sigmoid(logits)

    def train_model(
        self,
        data_loader: torch.utils.data.DataLoader,
        epochs: int,
        lr: float = 1e-3,
        device: str = "cpu",
        patience: int = 5,
    ) -> None:
        """Simple training loop with early stopping on validation loss."""
        self.to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        best_loss = float("inf")
        counter = 0
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for xb, yb in data_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                optimizer.zero_grad()
                preds = self.forward(xb).squeeze()
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(data_loader.dataset)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

    def evaluate(self, data_loader: torch.utils.data.DataLoader, device: str = "cpu") -> float:
        """Return average accuracy on a dataset."""
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in data_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                preds = self.forward(xb).squeeze() > 0.5
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / total


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetector"]
