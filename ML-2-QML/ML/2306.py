"""Hybrid fraud detection – classical implementation.

The class `FraudDetectionHybrid` builds a classical neural network that mimics
the photonic layer structure from the original `FraudDetection.py`.  It adds
clipping, a final classification head, and a simple training loop.  The
implementation is fully PyTorch‑based and can be used as a drop‑in
replacement for the legacy module while providing a richer interface for
experimentation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn, optim


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
    """Clip a scalar value to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single photonic‑style layer as a PyTorch module."""
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


class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection model – classical side.

    The network consists of a stack of photonic‑style layers followed by a
    small classification head.  Parameters can be clipped to keep the model
    stable during training.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        hidden_dim: int = 4,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.device = device
        # Build the feature extractor
        modules: list[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        # Classification head
        modules.append(nn.Linear(2, hidden_dim))
        modules.append(nn.ReLU())
        modules.append(nn.Linear(hidden_dim, 2))
        self.network = nn.Sequential(*modules).to(self.device)

        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.to(self.device))

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> None:
        """Simple training loop using cross‑entropy loss."""
        self.network.train()
        dataset = torch.utils.data.TensorDataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                logits = self.network(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Return class predictions for the input data."""
        self.network.eval()
        with torch.no_grad():
            logits = self.network(X.to(self.device))
        return logits.argmax(dim=1)

    def clip_parameters(self, bound: float = 5.0) -> None:
        """Clip all trainable parameters to keep them in a safe range."""
        for param in self.network.parameters():
            param.data.clamp_(-bound, bound)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
