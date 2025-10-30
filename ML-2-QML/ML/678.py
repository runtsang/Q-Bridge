"""Fraud detection using a purely classical neural network with training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FraudLayerParameters:
    """Parameters for a 2‑node fully‑connected layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(v: float, bound: float) -> float:
    return max(-bound, min(bound, v))


def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return Layer()


class FraudDetector(nn.Module):
    """Unified classical fraud‑detection model with training helpers.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (unclipped) layer.
    layers : Iterable[FraudLayerParameters]
        Subsequent layers (clipped to avoid exploding gradients).
    lr : float, default 1e-3
        Learning rate for the Adam optimiser.
    epochs : int, default 200
        Number of training epochs.
    batch_size : int, default 64
        Mini‑batch size.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 64,
    ) -> None:
        super().__init__()
        self.network = nn.Sequential(
            _layer_from_params(input_params, clip=False),
            *(_layer_from_params(l, clip=True) for l in layers),
            nn.Linear(2, 1),
        )
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self._optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self._criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.network(x).squeeze(-1)

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            self._optimizer.zero_grad()
            preds = self(xb)
            loss = self._criterion(preds, yb.float())
            loss.backward()
            self._optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        return epoch_loss / len(loader.dataset)

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Simple training loop with early‑stopping on validation loss."""
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        best_loss = float("inf")
        patience = 10
        counter = 0

        for epoch in range(self.epochs):
            loss = self._train_one_epoch(loader)
            if loss < best_loss:
                best_loss = loss
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return binary predictions (0 or 1)."""
        with torch.no_grad():
            logits = self(X)
            probs = torch.sigmoid(logits)
            return (probs > 0.5).long()

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> dict:
        """Return accuracy and loss on the given data."""
        self.eval()
        with torch.no_grad():
            logits = self(X)
            loss = self._criterion(logits, y.float()).item()
            preds = (torch.sigmoid(logits) > 0.5).long()
            acc = (preds == y).float().mean().item()
        return {"loss": loss, "accuracy": acc}


__all__ = ["FraudLayerParameters", "FraudDetector"]
