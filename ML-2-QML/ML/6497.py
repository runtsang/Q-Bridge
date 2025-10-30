"""Enhanced classical fraud‑detection model with Bayesian hyper‑parameter sampling.

The new `FraudDetectionModel` class extends the seed by providing:
* A Bayesian prior over the scale and shift buffers.
* A lightweight `train` method that accepts a DataLoader, optimizer, loss function and number of epochs.
* A `predict` method that returns probabilities via a sigmoid activation.

The model construction follows the original two‑layer architecture but allows clipping of parameters for stability.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Any

# --------------------------------------------------------------------------- #
# 1.  Hyper‑parameter definition
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# 2.  Model implementation
# --------------------------------------------------------------------------- #
class FraudDetectionModel(nn.Module):
    """A two‑layer neural network that mirrors a photonic circuit."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 clip: bool = True) -> None:
        super().__init__()
        self.input_params = input_params
        self.layers_params = list(layers)
        self.clip = clip
        self._build_model()

    def _build_model(self) -> None:
        modules: List[nn.Module] = [self._layer_from_params(self.input_params, clip=False)]
        modules.extend(self._layer_from_params(p, clip=self.clip) for p in self.layers_params)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules)

    def _layer_from_params(self, params: FraudLayerParameters, *, clip: bool) -> nn.Module:
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
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
                out = self.activation(self.linear(inputs))
                out = out * self.scale + self.shift
                return out

        return Layer()

    # --------------------------------------------------------------------- #
    # 3.  Bayesian hyper‑parameter sampling
    # --------------------------------------------------------------------- #
    def sample_hyperparameters(self, rng: np.random.Generator | None = None) -> None:
        """Draw scale and shift buffers from a standard normal prior."""
        rng = rng or np.random.default_rng()
        for buffer in self.parameters():
            shape = buffer.shape
            buffer.data.copy_(torch.tensor(rng.normal(size=shape), dtype=torch.float32))

    # --------------------------------------------------------------------- #
    # 4.  Training API
    # --------------------------------------------------------------------- #
    def train_model(self,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: nn.Module,
                    epochs: int = 10) -> None:
        self.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                preds = self(xb)
                loss = loss_fn(preds, yb)
                loss.backward()
                optimizer.step()

    # --------------------------------------------------------------------- #
    # 5.  Prediction
    # --------------------------------------------------------------------- #
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a probability in ``[0, 1]``."""
        self.eval()
        with torch.no_grad():
            logits = self(inputs)
            return torch.sigmoid(logits)

# --------------------------------------------------------------------------- #
# 6.  Public API
# --------------------------------------------------------------------------- #
__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
