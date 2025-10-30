"""Classical fraud detection model with extended functionality.

This module builds upon the original seed by adding:
- A reusable FraudDetectionModel class that encapsulates the sequential network.
- Utility methods for random parameter generation, parameter extraction, and simple
  training/evaluation loops.
- Support for clipping and optional scaling of weights and biases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, List, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

# -------------------------------------------------------------
# Parameter definition
# -------------------------------------------------------------
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

# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------
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

        def forward(self, inputs: Tensor) -> Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# -------------------------------------------------------------
# Main model class
# -------------------------------------------------------------
class FraudDetectionModel(nn.Module):
    """A configurable fraud‑detection network.

    The network is assembled from a list of :class:`FraudLayerParameters`.
    The first layer is treated as an “input” layer and its weights are not clipped,
    mirroring the photonic analogue. Subsequent layers are clipped to keep the
    parameters within a physically meaningful range.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Sequence[FraudLayerParameters],
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.input_params = input_params
        self.layers = list(layers)
        self.model = build_fraud_detection_program(input_params, self.layers)
        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    # ------------------------------------------------------------------
    # Parameter utilities
    # ------------------------------------------------------------------
    @staticmethod
    def random_parameters(num_layers: int, seed: int | None = None) -> Tuple[FraudLayerParameters, List[FraudLayerParameters]]:
        """Generate a random parameter set for a network of ``num_layers``."""
        rng = torch.manual_seed(seed) if seed is not None else torch.default_generator
        def rand_pair() -> Tuple[float, float]:
            return (float(rng.normal()), float(rng.normal()))
        input_params = FraudLayerParameters(
            bs_theta=float(rng.normal()),
            bs_phi=float(rng.normal()),
            phases=rand_pair(),
            squeeze_r=rand_pair(),
            squeeze_phi=rand_pair(),
            displacement_r=rand_pair(),
            displacement_phi=rand_pair(),
            kerr=rand_pair(),
        )
        layers = [
            FraudLayerParameters(
                bs_theta=float(rng.normal()),
                bs_phi=float(rng.normal()),
                phases=rand_pair(),
                squeeze_r=rand_pair(),
                squeeze_phi=rand_pair(),
                displacement_r=rand_pair(),
                displacement_phi=rand_pair(),
                kerr=rand_pair(),
            )
            for _ in range(num_layers)
        ]
        return input_params, layers

    @property
    def state_dict(self) -> Dict[str, Tensor]:
        """Return a flattened state dict of the underlying sequential model."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict: Dict[str, Tensor]) -> None:
        """Load a state dict into the underlying sequential model."""
        self.model.load_state_dict(state_dict)

    # ------------------------------------------------------------------
    # Simple training utilities
    # ------------------------------------------------------------------
    def train_on_loader(
        self,
        loader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        epochs: int = 10,
        device: torch.device | str | None = None,
    ) -> List[float]:
        """Train the model on a data loader and return a list of epoch losses."""
        losses: List[float] = []
        if device is not None:
            self.to(device)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch in loader:
                inputs, targets = batch
                if device is not None:
                    inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            epoch_loss /= len(loader.dataset)
            losses.append(epoch_loss)
        return losses

    def evaluate_on_loader(
        self,
        loader,
        criterion: nn.Module,
        device: torch.device | str | None = None,
    ) -> float:
        """Compute the loss over a validation set."""
        if device is not None:
            self.to(device)
        self.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in loader:
                inputs, targets = batch
                if device is not None:
                    inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
        return total_loss / len(loader.dataset)

__all__ = ["FraudLayerParameters", "FraudDetectionModel", "build_fraud_detection_program"]
