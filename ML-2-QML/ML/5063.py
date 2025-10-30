from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
from torch import nn
import numpy as np

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer (used for weight mapping)."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


class FCL(nn.Module):
    """Simple fully connected layer with a ``run`` method for compatibility."""
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()


class FraudDetectionModel(nn.Module):
    """
    Classical hybrid model that mirrors the photonic fraud‑detection circuit and
    incorporates a QCNN‑style feature extractor.  The architecture is:

        feature_extractor → photonic layers → final head → optional FCL

    The photonic layers are implemented with linear + tanh + scaling to emulate
    the continuous‑variable operations.  The model is fully differentiable and
    can be trained with standard PyTorch optimisers.
    """
    def __init__(self, num_features: int = 8, depth: int = 3, use_fcl: bool = True) -> None:
        super().__init__()
        # QCNN‑style feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 16), nn.Tanh(),
            nn.Linear(16, 16), nn.Tanh(),
            nn.Linear(16, 12), nn.Tanh(),
            nn.Linear(12, 8), nn.Tanh(),
            nn.Linear(8, 4), nn.Tanh(),
            nn.Linear(4, 4), nn.Tanh(),
        )
        # Photonic‑style layers
        self.photonic_layers = nn.ModuleList()
        for _ in range(depth):
            self.photonic_layers.append(self._photonic_layer(clip=True))
        # Final classification head
        self.head = nn.Linear(4, 1)
        # Optional fully‑connected layer
        self.fcl = FCL() if use_fcl else None

    def _photonic_layer(self, clip: bool) -> nn.Module:
        """Create a single photonic‑style block using a linear map, tanh, and scaling."""
        weight = torch.randn(2, 2)
        bias = torch.randn(2)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(2, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)

        activation = nn.Tanh()
        scale = torch.randn(2)
        shift = torch.randn(2)

        class Layer(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = linear
                self.activation = activation
                self.register_buffer("scale", scale)
                self.register_buffer("shift", shift)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                out = self.activation(self.linear(inputs))
                return out * self.scale + self.shift

        return Layer()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(inputs)
        # Reduce to 2‑dimensional space for photonic layers
        x = x[:, :2]
        for layer in self.photonic_layers:
            x = layer(x)
        out = self.head(x)
        if self.fcl is not None:
            out = out + torch.tensor(self.fcl.run(out.squeeze().tolist()), dtype=out.dtype)
        return out


__all__ = ["FraudLayerParameters", "FraudDetectionModel", "FCL"]
