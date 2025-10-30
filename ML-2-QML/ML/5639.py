"""Enhanced classical fraud‑detection model with quantum regularization.

This module implements a hybrid model that combines a classical neural
network with a quantum‑derived regularization term. The quantum term
encourages the first layer’s parameters to be consistent with a
photonic circuit representation, improving model interpretability
and providing a physics‑inspired penalty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits

# --------------------------------------------------------------------------- #
#  Data‑class describing a fully connected layer
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

# --------------------------------------------------------------------------- #
#  Helper functions
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _create_layer(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
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

def _build_classifier() -> nn.Sequential:
    """Small MLP head that follows the feature extractor."""
    return nn.Sequential(
        nn.Linear(2, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )

# --------------------------------------------------------------------------- #
#  Hybrid model
# --------------------------------------------------------------------------- #
class FraudDetectionHybrid(nn.Module):
    """Hybrid classical‑quantum fraud‑detection model.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (feature‑extractor) layer.
    layers : Iterable[FraudLayerParameters]
        Remaining layer parameters.
    quantum_module : Any
        Reference to the quantum module that provides a ``regularization_cost``.
    lambda_reg : float
        Weight of the quantum regularization term.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        quantum_module: object,
        lambda_reg: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_params = input_params
        self.feature_extractor = nn.Sequential(
            _create_layer(input_params, clip=False),
            *(_create_layer(l, clip=True) for l in layers),
        )
        self.classifier = _build_classifier()
        self.quantum_module = quantum_module
        self.lambda_reg = lambda_reg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits.squeeze(-1)

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the hybrid loss: BCE + λ * quantum regularization."""
        bce = binary_cross_entropy_with_logits(logits, targets.to(torch.float32))
        reg = self.quantum_module.regularization_cost(self.input_params)
        return bce + self.lambda_reg * reg

    def train_step(
        self,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Single training epoch."""
        self.train()
        metrics = {"loss": 0.0, "bce": 0.0, "reg": 0.0}
        for batch_idx, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()
            logits = self.forward(x)
            loss = self.loss(logits, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                bce = binary_cross_entropy_with_logits(logits, y.to(torch.float32))
                reg = self.quantum_module.regularization_cost(self.input_params)
                metrics["loss"] += loss.item()
                metrics["bce"] += bce.item()
                metrics["reg"] += reg.item()
        n = len(data_loader)
        return {k: v / n for k, v in metrics.items()}

    def evaluate(
        self,
        data_loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, float]:
        """Evaluation routine without gradients."""
        self.eval()
        metrics = {"loss": 0.0, "bce": 0.0, "reg": 0.0, "accuracy": 0.0}
        with torch.no_grad():
            for x, y in data_loader:
                logits = self.forward(x)
                loss = self.loss(logits, y)
                preds = torch.sigmoid(logits) > 0.5
                acc = (preds == y.bool()).float().mean()
                bce = binary_cross_entropy_with_logits(logits, y.to(torch.float32))
                reg = self.quantum_module.regularization_cost(self.input_params)

                metrics["loss"] += loss.item()
                metrics["bce"] += bce.item()
                metrics["reg"] += reg.item()
                metrics["accuracy"] += acc.item()
        n = len(data_loader)
        return {k: v / n for k, v in metrics.items()}

    def to(self, device: torch.device | str) -> None:
        super().to(device)
        self.device = torch.device(device)
        return self

    def __repr__(self) -> str:
        return f"<FraudDetectionHybrid device={self.device} lambda_reg={self.lambda_reg}>"
