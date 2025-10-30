"""Upgraded classical fraud detection model with residual connections, batch‑norm, and dropout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn

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
    use_batchnorm: bool = False

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
    modules: List[nn.Module] = [linear, activation]
    if params.use_batchnorm:
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.seq(inputs)
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [ _layer_from_params(input_params, clip=False) ]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FraudDetectionModel(nn.Module):
    """Hybrid‑residual fraud detection model.

    The model consists of a stack of fully‑connected layers with optional
    batch‑norm, dropout and residual connections.  The final layer outputs a
    single log‑it for binary classification.
    """
    def __init__(self, input_params: FraudLayerParameters, layers: List[FraudLayerParameters]) -> None:
        super().__init__()
        self._layers = [_layer_from_params(input_params, clip=False)]
        self._layers.extend(_layer_from_params(layer, clip=True) for layer in layers)
        self._final = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for layer in self._layers:
            out = layer(residual)
            residual = out + residual
        logits = self._final(residual)
        return logits

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        probs = torch.sigmoid(self.forward(x))
        return (probs > threshold).long()

    def evaluate(self, loader: Iterable, loss_fn=nn.BCEWithLogitsLoss()) -> dict[str, float]:
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in loader:
                logits = self.forward(data)
                loss = loss_fn(logits.squeeze(), target.float())
                total_loss += loss.item() * data.size(0)
                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds.squeeze() == target).sum().item()
                total += data.size(0)
        return {"loss": total_loss / total, "accuracy": correct / total}

    def train_model(
        self,
        loader: Iterable,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        loss_fn=nn.BCEWithLogitsLoss(),
        verbose: bool = False,
    ) -> None:
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for data, target in loader:
                optimizer.zero_grad()
                logits = self.forward(data)
                loss = loss_fn(logits.squeeze(), target.float())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * data.size(0)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss / len(loader.dataset):.4f}")

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionModel"]
