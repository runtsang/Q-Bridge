"""Classical fraud‑detection model mirroring the photonic architecture.

The model is a `torch.nn.Module` that reproduces the layered
structure of the seed, but adds dropout, weight clipping, an
early‑stopping training loop and utilities for checkpointing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Sequence, Iterator

import torch
from torch import nn
from torch.utils.data import DataLoader


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


def _clip(value: float, bound: float) -> float:
    """Clips a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Builds a single linear‑activation block from the supplied parameters."""
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


class FraudDetectionHybrid(nn.Module):
    """Classical fraud‑detection model with dropout, clipping and early stopping."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout_p: float = 0.2,
        clip: bool = True,
    ) -> None:
        super().__init__()
        self.clip = clip
        modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=self.clip) for layer in layers)
        modules.append(nn.Linear(2, 1))
        self.network = nn.Sequential(*modules)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(x))

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 1e-3,
        early_stop_patience: int = 5,
    ) -> List[float]:
        """Train using binary cross‑entropy with early stopping on validation loss."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        best_val = float("inf")
        patience = 0
        val_losses: List[float] = []

        for epoch in range(epochs):
            self.train()
            for x, y in train_loader:
                optimizer.zero_grad()
                logits = self(x)
                loss = criterion(logits.squeeze(), y.float())
                loss.backward()
                optimizer.step()

            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_loader:
                    logits = self(x)
                    val_loss += criterion(logits.squeeze(), y.float()).item()
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.state_dict(), "best_fraud_model.pt")
                patience = 0
            else:
                patience += 1

            if patience >= early_stop_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.load_state_dict(torch.load("best_fraud_model.pt"))
        return val_losses

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return binary predictions."""
        self.eval()
        with torch.no_grad():
            return (self(x) > 0.5).float()

    @staticmethod
    def load_from_checkpoint(path: str) -> "FraudDetectionHybrid":
        """Instantiate from a saved state_dict."""
        ckpt = torch.load(path)
        # Create a minimal skeleton; user must re‑initialize layers.
        model = FraudDetectionHybrid(
            input_params=FraudLayerParameters(0, 0, (0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)),
            layers=[],
        )
        model.load_state_dict(ckpt)
        return model


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
