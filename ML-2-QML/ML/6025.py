"""Hybrid fraud detection model combining classical neural network and quantum kernel layer.

The model architecture mirrors the photonic fraud detection circuit but augments
the feature extraction with a quantum kernel module implemented with TorchQuantum.
The quantum kernel operates on a set of support vectors and returns a similarity
matrix that is concatenated with the classical features before the final
classification layer.

The module is fully importable and can be used as a drop‑in replacement for the
original `build_fraud_detection_program` function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, List

import torch
from torch import nn

# Optional import of the quantum kernel module from the QML side.
try:
    from.qml import QuantumKernelLayer  # type: ignore
except Exception:
    # Fallback: simple RBF kernel implemented with NumPy.
    class QuantumKernelLayer(nn.Module):
        def __init__(self, support_vectors: torch.Tensor, gamma: float = 1.0):
            super().__init__()
            self.support_vectors = support_vectors
            self.gamma = gamma

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            diff = x[:, None, :] - self.support_vectors[None, :, :]
            dist2 = torch.sum(diff * diff, dim=-1)
            return torch.exp(-self.gamma * dist2)

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

class FraudDetectionHybrid(nn.Module):
    """
    Classical neural network that optionally appends a quantum kernel layer.
    The network architecture follows the photonic fraud detection circuit but
    adds a quantum kernel module that computes similarity to a set of support
    vectors. The final output is a binary classification score.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        support_vectors: torch.Tensor | None = None,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
        self.classical = nn.Sequential(*modules)

        if support_vectors is not None:
            self.kernel_layer = QuantumKernelLayer(
                support_vectors=support_vectors, gamma=kernel_gamma
            )
            self.concat = nn.Linear(2 + support_vectors.shape[0], 1)
        else:
            self.kernel_layer = None
            self.concat = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        classical_feat = self.classical(x)
        if self.kernel_layer is not None:
            kernel_feat = self.kernel_layer(classical_feat)
            out = torch.cat([classical_feat, kernel_feat], dim=-1)
        else:
            out = classical_feat
        return self.concat(out)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
    support_vectors: torch.Tensor | None = None,
    kernel_gamma: float = 1.0,
) -> FraudDetectionHybrid:
    """
    Construct a hybrid fraud detection model.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (input) layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for subsequent layers.
    support_vectors : torch.Tensor, optional
        Support vectors for the quantum kernel. If ``None`` the model will
        operate purely classically.
    kernel_gamma : float, default 1.0
        Gaussian kernel width when a fallback RBF kernel is used.

    Returns
    -------
    FraudDetectionHybrid
        The hybrid model ready for training.
    """
    return FraudDetectionHybrid(
        input_params=input_params,
        layers=layers,
        support_vectors=support_vectors,
        kernel_gamma=kernel_gamma,
    )

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionHybrid",
    "QuantumKernelLayer",
]
