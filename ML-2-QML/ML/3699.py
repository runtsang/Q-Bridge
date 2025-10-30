"""Hybrid fraud detection model that couples classical feature extraction with a dual‑quantum backend.

The class `FraudDetectionHybrid` inherits from `torch.nn.Module`.  It builds a small feed‑forward extractor that emits parameters for a photonic circuit.  These parameters are fed to a quantum module implemented in :mod:`FraudDetection__gen346_qml`, which returns a 2‑dimensional feature vector (photonic expectation + qubit expectation).  A final linear layer maps the 2‑dimensional quantum representation to a scalar fraud score.

The implementation keeps the quantum logic in a separate module so that the ML code remains purely classical.  The quantum module is lazily imported to avoid circular dependencies.
"""

from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

# --- Classical layer description (used by both sides) ---------------------------------
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

# --- Utility helpers ---------------------------------------------------------------
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

# --- Classical feature extractor ----------------------------------------------------
class FraudFeatureExtractor(nn.Module):
    """Feed‑forward network that emits parameters for the photonic circuit."""
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        # Each photonic layer needs 8 parameters; we output 8*(num_layers+1)
        self.output_dim = 8 * (self.num_layers + 1)
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, self.output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

# --- Full hybrid model ---------------------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """Combines a classical extractor with a quantum backend."""
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.extractor = FraudFeatureExtractor(num_layers)
        self.final = nn.Linear(2, 1)               # 2 quantum features -> scalar score
        # Lazy import of the quantum module
        import FraudDetection__gen346_qml as qm
        self.quantum_module = qm.FraudDetectionHybrid(num_layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        params = self.extractor(inputs)            # shape: (batch, 8*(L+1))
        quantum_feats = self.quantum_module(params)  # shape: (batch, 2)
        return self.final(quantum_feats)

__all__ = ["FraudDetectionHybrid", "FraudLayerParameters"]
