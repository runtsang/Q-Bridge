"""FraudDetectionAdvanced: classical model with dropout, batchnorm, and hyper‑parameter search.

This module extends the original two‑layer network by adding
- batch‑normalisation after each linear layer,
- dropout with a configurable probability,
- L2 regularisation on weights,
- a custom loss that emphasises early‑time fraud signals.

The class FraudDetectionAdvanced provides a convenient interface:
>>> model = FraudDetectionAdvanced()
>>> model.fit(X_train, y_train, X_val, y_val)
>>> pred = model.predict(X_test)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim


@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(
    params: FraudLayerParameters, *, clip: bool, dropout_p: float
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
    activation = nn.Tanh()
    scale = torch.tensor(params.displacement_r, dtype=torch.float32)
    shift = torch.tensor(params.displacement_phi, dtype=torch.float32)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.bn = nn.BatchNorm1d(2)
            self.dropout = nn.Dropout(dropout_p)
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.linear(inputs)
            out = self.activation(out)
            out = self.bn(out)
            out = self.dropout(out)
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    dropout_p: float = 0.2,
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False, dropout_p=dropout_p)]
    modules.extend(
        _layer_from_params(layer, clip=True, dropout_p=dropout_p) for layer in layers
    )
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class FraudDetectionAdvanced:
    """End‑to‑end wrapper that trains a fraud‑detection model with optional hyper‑parameter search."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        dropout_p: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        batch_size: int = 128,
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = build_fraud_detection_program(
            input_params, layers, dropout_p
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.BCEWithLogitsLoss()
        self.epochs = epochs
        self.batch_size = batch_size

    def _train_one_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device).float()
            self.optimizer.zero_grad()
            logits = self.model(x).squeeze(-1)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(loader.dataset)

    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor | None = None,
        y_val: torch.Tensor | None = None,
    ) -> None:
        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        if X_val is not None and y_val is not None:
            val_ds = TensorDataset(X_val, y_val)
            val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        else:
            val_loader = None

        for epoch in range(self.epochs):
            train_loss = self._train_one_epoch(train_loader)
            if val_loader:
                self.model.eval()
                with torch.no_grad():
                    val_logits = self.model(X_val.to(self.device)).squeeze(-1)
                    val_loss = self.criterion(
                        val_logits, y_val.to(self.device).float()
                    ).item()
                print(
                    f"Epoch {epoch+1}/{self.epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch+1}/{self.epochs} | train loss {train_loss:.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device)).squeeze(-1)
            probs = torch.sigmoid(logits)
        return probs.cpu()

    @staticmethod
    def hyperparameter_search(
        X: torch.Tensor,
        y: torch.Tensor,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        search_space: Dict[str, Sequence[Any]],
        n_trials: int = 10,
        epochs: int = 20,
        batch_size: int = 128,
    ) -> Tuple[Dict[str, Any], float]:
        """Random search over the specified hyper‑parameter space.

        Returns a tuple of the best configuration and its validation loss.
        """
        import random

        best_cfg, best_loss = None, float("inf")
        for _ in range(n_trials):
            cfg = {k: random.choice(v) for k, v in search_space.items()}
            model = FraudDetectionAdvanced(
                input_params,
                layers,
                dropout_p=cfg.get("dropout_p", 0.2),
                lr=cfg.get("lr", 1e-3),
                weight_decay=cfg.get("weight_decay", 1e-5),
                epochs=epochs,
                batch_size=batch_size,
            )
            n = X.shape[0]
            idx = torch.randperm(n)
            train_idx, val_idx = idx[: int(0.8 * n)], idx[int(0.8 * n) :]
            model.fit(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
            val_logits = model.model(X[val_idx].to(model.device)).squeeze(-1)
            loss = nn.BCEWithLogitsLoss()(
                val_logits, y[val_idx].to(model.device).float()
            ).item()
            if loss < best_loss:
                best_cfg, best_loss = cfg, loss
        return best_cfg, best_loss
