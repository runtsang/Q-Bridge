from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

# --------------------------------------------------------------------------- #
# 1. Classical quanvolution filter
# --------------------------------------------------------------------------- #
class QuanvolutionFilter(nn.Module):
    """Simple 2×2 stride‑2 convolution that reshapes the output into a flat feature vector."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)  # flatten per sample

# --------------------------------------------------------------------------- #
# 2. Photonic‑style fraud layer (classical implementation)
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

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Construct a sequential PyTorch model that mirrors the photonic fraud detection stack."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 3. Classical classifier construction (mirrors the quantum ansatz)
# --------------------------------------------------------------------------- #
def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], Sequence[int]]:
    """Create a feed‑forward classifier with metadata similar to the quantum version."""
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []

    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features

    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())

    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

# --------------------------------------------------------------------------- #
# 4. Unified hybrid classifier
# --------------------------------------------------------------------------- #
class HybridQuanvolutionGraphClassifier(nn.Module):
    """
    Combines:
      * a classical quanvolution filter,
      * a fraud‑detection style sequential block,
      * a variational classifier network.
    The class exposes a single ``forward`` method that can be used for training or inference.
    """
    def __init__(
        self,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layers_params: Sequence[FraudLayerParameters] | None = None,
        classifier_depth: int = 2,
    ) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()

        # Fraud detection block (if parameters are provided)
        if fraud_input_params is not None and fraud_layers_params is not None:
            self.fraud_block = build_fraud_detection_program(
                fraud_input_params, fraud_layers_params
            )
        else:
            self.fraud_block = nn.Identity()

        # Classifier head
        self.classifier, _, _, _ = build_classifier_circuit(
            num_features=4 * 14 * 14, depth=classifier_depth
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, 2).
        """
        features = self.qfilter(x)                     # (batch, 4*14*14)
        fraud_out = self.fraud_block(features)          # (batch, 2)
        logits = self.classifier(fraud_out)             # (batch, 2)
        return F.log_softmax(logits, dim=-1)

__all__ = [
    "QuanvolutionFilter",
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "build_classifier_circuit",
    "HybridQuanvolutionGraphClassifier",
]
