"""Enhanced fraud detection model with training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import torch
from torch import nn, optim
import torch.nn.functional as F


@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

    def to_tensor(self) -> torch.Tensor:
        """Flatten parameters into a single tensor for gradient tracking."""
        return torch.tensor(
            [
                self.bs_theta,
                self.bs_phi,
                *self.phases,
                *self.squeeze_r,
                *self.squeeze_phi,
                *self.displacement_r,
                *self.displacement_phi,
                *self.kerr,
            ],
            dtype=torch.float32,
        )


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


class FraudDetectionHybridModel(nn.Module):
    """Full classical fraud detection network with training helpers."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.to(device)
        modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        early_stopping_patience: int = 5,
        verbose: bool = True,
    ) -> None:
        """Train the model with optional early stopping."""
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=False
        )
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self(batch_x).squeeze()
                loss = F.mse_loss(preds, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)

            epoch_loss /= len(train_loader.dataset)

            val_loss = None
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        preds = self(batch_x).squeeze()
                        val_loss += F.mse_loss(preds, batch_y, reduction="sum").item()
                    val_loss /= len(val_loader.dataset)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

            if verbose:
                msg = f"Epoch {epoch:02d} | Train loss: {epoch_loss:.4f}"
                if val_loader is not None:
                    msg += f" | Val loss: {val_loss:.4f}"
                print(msg)

            if val_loader is not None and patience_counter >= early_stopping_patience:
                if verbose:
                    print("Early stopping triggered.")
                break

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions for input tensor."""
        self.eval()
        with torch.no_grad():
            return self(x).squeeze()


__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybridModel",
]
