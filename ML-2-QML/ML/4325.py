import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

# ----------------------------------------------------------------------
# Classical fraud‑detection inspired layer construction
# ----------------------------------------------------------------------
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


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    # Build a linear layer whose weights encode the beam‑splitter and squeezing
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a PyTorch Sequential mirroring the photonic fraud‑detection stack."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))  # final read‑out
    return nn.Sequential(*modules)


# ----------------------------------------------------------------------
# Classical regression dataset (from QuantumRegression.py)
# ----------------------------------------------------------------------
def generate_superposition_data(num_features: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample states of the form cos(θ)|0…0⟩ + e^{iφ} sin(θ)|1…1⟩."""
    x = np.random.uniform(-1.0, 1.0, size=(samples, num_features)).astype(np.float32)
    angles = x.sum(axis=1)
    y = np.sin(angles) + 0.1 * np.cos(2 * angles)
    return x, y.astype(np.float32)


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, samples: int, num_features: int):
        self.features, self.labels = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.features)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:  # type: ignore[override]
        return {
            "states": torch.tensor(self.features[idx], dtype=torch.float32),
            "target": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ----------------------------------------------------------------------
# Hybrid fully‑connected layer (classical)
# ----------------------------------------------------------------------
class HybridFCL(nn.Module):
    """
    A classical neural network that emulates the fraud‑detection photonic
    pipeline while being fully differentiable in PyTorch.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Sequence[FraudLayerParameters],
    ):
        super().__init__()
        self.model = build_fraud_detection_program(input_params, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Convenience wrapper: accepts a flat list of parameters,
        turns it into a 2‑dim tensor, and returns the network output.
        """
        params = np.array(list(thetas), dtype=np.float32).reshape(1, -1)
        return self.forward(torch.tensor(params)).detach().numpy()


__all__ = [
    "HybridFCL",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "generate_superposition_data",
    "RegressionDataset",
]
