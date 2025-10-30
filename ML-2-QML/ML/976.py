# FraudDetectionHybrid: Classical PyTorch implementation of the fraud detection model.
# This module extends the original seed by adding a lightweight training pipeline,
# dropout and batch‑normalisation layers, and a convenient ``predict`` method.
# The design keeps the original ``FraudLayerParameters`` dataclass for
# parameter sharing, but adds a ``dropout`` and ``batch_norm`` field to
# simplify construction of arbitrarily deep networks.

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn, optim
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
    # Additional hyper‑parameters for the classical network
    dropout: float = 0.0
    batch_norm: bool = False


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
            self.bn = nn.BatchNorm1d(2) if params.batch_norm else None
            self.drop = nn.Dropout(p=params.dropout) if params.dropout > 0.0 else None

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            if self.bn is not None:
                outputs = self.bn(outputs)
            if self.drop is not None:
                outputs = self.drop(outputs)
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()


class FraudDetectionHybrid:
    """Hybrid fraud‑detection model that can be trained classically.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer (input layer).
    layers : Iterable[FraudLayerParameters]
        Parameters for all subsequent hidden layers.
    device : str, optional
        Device on which the model will be executed.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device: str = "cpu",
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.device = device

        modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in self.layers)
        modules.append(nn.Linear(2, 1))
        self.model = nn.Sequential(*modules).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(inputs)

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return fraud probability after sigmoid."""
        with torch.no_grad():
            logits = self.forward(inputs)
            probs = torch.sigmoid(logits)
        return probs

    def train(
        self,
        data_loader: DataLoader,
        epochs: int = 10,
        verbose: bool = True,
    ) -> None:
        """Simple training loop for the classical model."""
        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = self.loss_fn(logits.squeeze(), y)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * x.size(0)
            epoch_loss /= len(data_loader.dataset)
            if verbose:
                print(f"Epoch {epoch:02d} – loss: {epoch_loss:.4f}")

    def synthetic_dataset(self, n_samples: int = 1000) -> DataLoader:
        """Generate a toy dataset for demonstration purposes."""
        torch.manual_seed(42)
        X = torch.randn(n_samples, 2)
        # Simple rule: fraud if x1 + x2 > 0
        y = (X.sum(dim=1) > 0).float()
        dataset = TensorDataset(X, y)
        return DataLoader(dataset, batch_size=32, shuffle=True)

    def to_quantum_params(self) -> List[FraudLayerParameters]:
        """Return the list of parameters that can be fed to the quantum module."""
        return [self.input_params] + self.layers

    def __repr__(self) -> str:
        return f"<FraudDetectionHybrid device={self.device} layers={len(self.layers)}>"

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
