"""Enhanced classical fraud detection model with training utilities and advanced regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


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


class FraudDetectionPipeline(nn.Module):
    """A full-fledged fraud‑detection pipeline that can be trained end‑to‑end.

    The pipeline exposes:
      * a `forward` method that returns logits
      * a `train_step` helper that performs a single SGD update
      * an `evaluate` helper that returns accuracy and loss
      * utilities to save and restore checkpoints
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: float | None = None,
    ) -> None:
        super().__init__()
        self.model = build_fraud_detection_program(input_params, layers)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        logits = self.forward(x)
        loss = self.loss_fn(logits.squeeze(), y.float())
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss.item()

    def evaluate(self, dataloader: DataLoader) -> tuple[float, float]:
        self.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in dataloader:
                logits = self.forward(xb)
                loss = self.loss_fn(logits.squeeze(), yb.float())
                total_loss += loss.item() * xb.size(0)
                preds = torch.sigmoid(logits) > 0.5
                correct += (preds.squeeze() == yb).sum().item()
                total += xb.size(0)
        return total_loss / total, correct / total

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str) -> None:
        self.load_state_dict(torch.load(path))


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionPipeline"]
