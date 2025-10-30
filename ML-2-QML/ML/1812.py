"""Hybrid fraud‑detection model with a classical head and optional early‑stopping.

The classical part mirrors the photonic circuit but adds a trainable
dense block that consumes the two‑dimensional output of the
variational layers.  Early‑stopping is implemented by monitoring a
validation loss that is reset each epoch; when it does not improve
after ``patience`` epochs the optimizer is frozen.

The module is fully Torch‑compatible and can be used with any
DataLoader or Trainer.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

# --------------------------------------------------------------------------- #
#  Classical block – keeps the same interface as the original seed
# --------------------------------------------------------------------------- #
@dataclass
class FraudLayerParameters:
    """Parameters defining a single layer in the classical analog of
    a photonic circuit.  The shape matches the input‑parameter
    seed, but we augment the head with *all* two‑dimensional
    (x,y) outputs from each layer.
    """
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the interval [−bound, bound]."""
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Construct a single linear‑tanh‑linear block from the
    provided parameters.
    """
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

class FraudDetectionModel(nn.Module):
    """A classical fraud‑detection model that emulates the photonic
    circuit and appends a trainable dense head.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        *,
        head_hidden_dim: int = 8,
        head_dropout: float = 0.0,
        patience: int = 5,
    ) -> None:
        super().__init__()
        self.patience = patience
        self._early_stop_counter = 0
        self._stop_training = False

        # Build the sequence of feature‑extracting layers
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(p, clip=True) for p in layer_params)

        # Trainable head
        modules.append(nn.Linear(2, head_hidden_dim))
        modules.append(nn.ReLU())
        if head_dropout > 0.0:
            modules.append(nn.Dropout(head_dropout))
        modules.append(nn.Linear(head_hidden_dim, 1))

        self.features = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the feature extractor and dense head."""
        return self.features(x)

    def monitor(self, val_loss: float) -> None:
        """Update early‑stopping counter based on validation loss.

        This method should be called once per validation epoch.
        """
        if val_loss is None:
            return
        if not hasattr(self, "_best_val_loss"):
            self._best_val_loss = val_loss
            self._early_stop_counter = 0
        elif val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._early_stop_counter = 0
        else:
            self._early_stop_counter += 1
            if self._early_stop_counter >= self.patience:
                self._stop_training = True

    @property
    def stop_training(self) -> bool:
        """Return whether training should be halted."""
        return self._stop_training

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionModel",
]
