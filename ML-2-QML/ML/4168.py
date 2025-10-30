"""Combined classical convolution and fraud‑detection inspired network."""
from __future__ import annotations

import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully‑connected layer in the classical model."""
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

class ConvFilter(nn.Module):
    """2‑D convolution filter with sigmoid activation and a threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        # Two output channels so that the fraud‑detection layers receive a 2‑dim vector.
        self.conv = nn.Conv2d(1, 2, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        # Reduce spatial dimensions by mean and shift by the threshold
        return torch.sigmoid(logits - self.threshold).mean(dim=(2, 3))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single fraud‑detection style layer from parameters."""
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

        def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            out = self.activation(self.linear(inp))
            return out * self.scale + self.shift

    return Layer()

def build_combined_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Build a sequential network mirroring the fraud‑detection structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class CombinedConvNet(nn.Module):
    """End‑to‑end network that applies a convolution filter, fraud‑style layers,
    and a linear head producing a vector for the quantum expectation."""
    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        fraud_params: Iterable[FraudLayerParameters] = (),
        n_qubits: int = 2,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size, threshold)
        self.fraud_net = build_combined_program(fraud_params[0], fraud_params[1:])
        self.linear = nn.Linear(1, n_qubits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a batch of angle vectors for the quantum head."""
        x = self.conv(x)          # shape (batch, 2)
        x = self.fraud_net(x)     # shape (batch, 1)
        return self.linear(x)     # shape (batch, n_qubits)

__all__ = [
    "ConvFilter",
    "FraudLayerParameters",
    "build_combined_program",
    "CombinedConvNet",
]
