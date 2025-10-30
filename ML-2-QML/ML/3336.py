"""Hybrid fraud‑detection model combining photonic‑style classical layers with a quantum‑inspired encoder.

The module exposes a classical neural network that imitates the photonic circuit from the original FraudDetection seed. It also re‑uses the superposition data generator from QuantumRegression to provide a synthetic fraud dataset.  The model is fully classical and ready for integration into standard PyTorch workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn

# --------------------------------------------------------------------------- #
# Parameter container – identical to the photonic seed but extended with a
# quantum‑specific field for future use.
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
    # quantum‑specific field (not used in the classical model but kept for
    # consistency with the QML counterpart)
    num_qubits: int = 2


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    """Clamp a scalar to the range [-bound, bound]."""
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single photonic‑style layer as an nn.Module."""
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
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
    """Construct a sequential PyTorch model mirroring the layered photonic design."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# --------------------------------------------------------------------------- #
# Synthetic dataset – re‑used from the QuantumRegression seed
# --------------------------------------------------------------------------- #
def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a synthetic dataset of superposition states with a sinusoid label."""
    import numpy as np

    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class FraudDataset(torch.utils.data.Dataset):
    """Dataset wrapping the synthetic superposition data for fraud detection."""

    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "states": self.features[index],
            "target": self.labels[index],
        }


# --------------------------------------------------------------------------- #
# Example model – a simple wrapper around the sequential construction
# --------------------------------------------------------------------------- #
class FraudDetectionHybridModel(nn.Module):
    """A classical model that reproduces the photonic fraud‑detection circuit."""

    def __init__(self, input_params: FraudLayerParameters, hidden_params: List[FraudLayerParameters]):
        super().__init__()
        self.net = build_fraud_detection_program(input_params, hidden_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "generate_superposition_data",
    "FraudDataset",
    "FraudDetectionHybridModel",
]
