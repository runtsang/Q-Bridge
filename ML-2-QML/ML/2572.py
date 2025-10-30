"""Hybrid fraud‑detection model with photonic‑inspired layers and a quantum kernel feature map.

The module re‑implements the classical photonic circuit as a PyTorch `nn.Module`, then augments the
feature vector with a quantum kernel evaluation.  The final classifier is a linear layer that
produces a fraud probability.  The design follows the “combination” scaling paradigm: classical
layers for efficient forward passes, quantum kernel for expressive similarity, and a shared
parameter interface for easy experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn

# --------------------------------------------------------------------------- #
#  Photonic‑inspired layer construction (adapted from the original seed)
# --------------------------------------------------------------------------- #

@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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
    weight = torch.tensor(
        [[params.bs_theta, params.bs_phi],
         [params.squeeze_r[0], params.squeeze_r[1]]],
        dtype=torch.float32
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
    """Create a sequential PyTorch model mirroring the photonic layer stack."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
#  Quantum kernel integration
# --------------------------------------------------------------------------- #

# The quantum kernel is defined in a separate module (QuantumKernelMethod.py).
# Import it lazily to avoid circular dependencies.
try:
    from.QuantumKernelMethod import Kernel as QuantumKernel
except Exception as exc:  # pragma: no cover
    raise ImportError("QuantumKernelMethod module required for quantum kernel features.") from exc

class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model.

    Parameters
    ----------
    base_params : FraudLayerParameters
        Parameters for the first (input) photonic layer.
    layer_params : Iterable[FraudLayerParameters]
        Parameters for the subsequent photonic layers.
    kernel_refs : Sequence[torch.Tensor]
        Reference vectors used by the quantum kernel to compute similarity.
    """

    def __init__(
        self,
        base_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        kernel_refs: Sequence[torch.Tensor],
    ) -> None:
        super().__init__()
        self.base_net = build_fraud_detection_program(base_params, layer_params)
        self.kernel = QuantumKernel()
        self.kernel_refs = torch.stack(kernel_refs)  # shape: (N, D)
        # Final classifier: concatenates photonic output and quantum kernel vector
        self.classifier = nn.Linear(1 + len(kernel_refs), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 2).

        Returns
        -------
        torch.Tensor
            Fraud probability in (0, 1) after sigmoid.
        """
        # Photonic feature
        photonic_out = self.base_net(x)  # shape: (batch, 1)

        # Quantum kernel feature: similarity to each reference vector
        # The kernel returns a scalar per reference; stack into a vector.
        kernel_feats = torch.stack(
            [self.kernel(x, ref.unsqueeze(0)) for ref in self.kernel_refs],
            dim=1,
        )  # shape: (batch, N)

        # Concatenate features
        features = torch.cat([photonic_out, kernel_feats], dim=1)

        logits = self.classifier(features)
        return torch.sigmoid(logits)

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
]
