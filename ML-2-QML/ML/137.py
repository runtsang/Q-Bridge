"""Classical fraud detection model with enhanced parameter handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Dict, Any
import torch
from torch import nn
import numpy as np
import random

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
    clip: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if name in ("phases", "squeeze_r", "squeeze_phi",
                        "displacement_r", "displacement_phi", "kerr"):
                if not isinstance(value, (tuple, list)) or len(value)!= 2:
                    raise ValueError(f"{name} must be a 2-tuple of floats")
        if self.clip:
            self._clip_params()

    def _clip_params(self) -> None:
        bound = 5.0
        for field_name in ("bs_theta", "bs_phi", "phases", "squeeze_r",
                           "squeeze_phi", "displacement_r", "displacement_phi"):
            value = getattr(self, field_name)
            if isinstance(value, (tuple, list)):
                clipped = tuple(max(-bound, min(bound, v)) for v in value)
                setattr(self, field_name, clipped)
            else:
                setattr(self, field_name,
                        max(-bound, min(bound, value)))
        self.kerr = tuple(max(-1.0, min(1.0, k)) for k in self.kerr)

    @classmethod
    def random(cls, clip: bool = False) -> "FraudLayerParameters":
        """Generate a random parameter set for debugging or initialization."""
        rng = random.Random()
        return cls(
            bs_theta=rng.uniform(-np.pi, np.pi),
            bs_phi=rng.uniform(-np.pi, np.pi),
            phases=(rng.uniform(-np.pi, np.pi),
                    rng.uniform(-np.pi, np.pi)),
            squeeze_r=(rng.uniform(0, 2), rng.uniform(0, 2)),
            squeeze_phi=(rng.uniform(-np.pi, np.pi),
                         rng.uniform(-np.pi, np.pi)),
            displacement_r=(rng.uniform(0, 2), rng.uniform(0, 2)),
            displacement_phi=(rng.uniform(-np.pi, np.pi),
                              rng.uniform(-np.pi, np.pi)),
            kerr=(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "bs_theta": self.bs_theta,
            "bs_phi": self.bs_phi,
            "phases": self.phases,
            "squeeze_r": self.squeeze_r,
            "squeeze_phi": self.squeeze_phi,
            "displacement_r": self.displacement_r,
            "displacement_phi": self.displacement_phi,
            "kerr": self.kerr,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FraudLayerParameters":
        """Deserialize from a dict."""
        return cls(**data)

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

def build_fraud_detection_model(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

__all__ = ["FraudLayerParameters", "build_fraud_detection_model"]
