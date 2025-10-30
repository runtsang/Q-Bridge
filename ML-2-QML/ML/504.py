"""Enhanced classical fraud detection model with training pipeline and metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchmetrics
import numpy as np

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

    def clip(self, bound: float = 5.0) -> "FraudLayerParameters":
        """Return a new instance with parameters clipped to [-bound, bound]."""
        return FraudLayerParameters(
            bs_theta=np.clip(self.bs_theta, -bound, bound),
            bs_phi=np.clip(self.bs_phi, -bound, bound),
            phases=tuple(np.clip(np.array(self.phases), -bound, bound)),
            squeeze_r=tuple(np.clip(np.array(self.squeeze_r), -bound, bound)),
            squeeze_phi=tuple(np.clip(np.array(self.squeeze_phi), -bound, bound)),
            displacement_r=tuple(np.clip(np.array(self.displacement_r), -bound, bound)),
            displacement_phi=tuple(np.clip(np.array(self.displacement_phi), -bound, bound)),
            kerr=tuple(np.clip(np.array(self.kerr), -bound, bound)),
        )

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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class FraudDetection(nn.Module):
    """Hybrid fraud detection model that can be trained end‑to‑end."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        clip: bool = True,
    ) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=clip) for layer in layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(x))

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _create_dataloaders(
        X: np.ndarray, y: np.ndarray, batch_size: int = 64, val_split: float = 0.2
    ) -> Tuple[DataLoader, DataLoader]:
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                                torch.tensor(y, dtype=torch.float32).unsqueeze(1))
        train_len = int(len(dataset) * (1 - val_split))
        val_len = len(dataset) - train_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size)
        return train_loader, val_loader

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 64,
        patience: int = 5,
        device: str = "cpu",
    ) -> List[float]:
        """Train the model with early stopping."""
        self.to(device)
        train_loader, val_loader = self._create_dataloaders(X, y, batch_size)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        best_val_loss = float("inf")
        epochs_no_improve = 0
        val_losses: List[float] = []

        for epoch in range(epochs):
            self.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = self(xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()

            val_loss = self.evaluate(val_loader, criterion, device)
            val_losses.append(val_loss.item())
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                break

        return val_losses

    def evaluate(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        device: str = "cpu",
    ) -> torch.Tensor:
        self.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = self(xb)
                loss = criterion(preds, yb)
                losses.append(loss)
        return torch.stack(losses).mean()

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5,
        device: str = "cpu",
    ) -> np.ndarray:
        self.to(device)
        self.eval()
        with torch.no_grad():
            preds = self(torch.tensor(X, dtype=torch.float32).to(device))
        return (preds.cpu().numpy() >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------
    def metrics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        device: str = "cpu",
    ) -> dict:
        preds = self.predict(X, device=device)
        acc = torchmetrics.functional.accuracy(torch.tensor(preds), torch.tensor(y))
        auc = torchmetrics.functional.auroc(
            torch.tensor(preds, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        )
        return {"accuracy": acc.item(), "auc": auc.item()}

__all__ = ["FraudLayerParameters", "FraudDetection"]
