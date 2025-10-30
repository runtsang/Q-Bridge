"""Hybrid quanvolution + fraud detection classifier with quantum filter.

This module implements a quantum version of the classical quanvolution filter
and pairs it with a fraud‑detection style classical head.  It is fully
compatible with the original Quanvolution.py API while exposing the quantum
core.

Classes
-------
HybridQuanvolutionFraudClassifier
    Quantum‑enabled model that combines a random quantum kernel applied
    to 2×2 image patches with a parameterised linear head.
QuanvolutionFilter, QuanvolutionClassifier
    Backwards‑compatible wrappers around the legacy classical filter
    and classifier, but implemented with the quantum filter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchquantum as tq
from dataclasses import dataclass
from typing import Iterable, Tuple

# Fraud‑detection style head ----------------------------------------------

@dataclass
class FraudLayerParameters:
    """Parameters for a single fraud‑detection style linear layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]

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

def build_fraud_detection_head(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the fraud‑detection layers."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# Quantum quanvolution filter -----------------------------------------------

class QuantumQuanvolutionFilter(tq.QuantumModule):
    """Apply a random two‑qubit quantum kernel to 2×2 image patches."""
    def __init__(self) -> None:
        super().__init__()
        self.n_wires = 4
        self.encoder = tq.GeneralEncoder(
            [
                {"input_idx": [0], "func": "ry", "wires": [0]},
                {"input_idx": [1], "func": "ry", "wires": [1]},
                {"input_idx": [2], "func": "ry", "wires": [2]},
                {"input_idx": [3], "func": "ry", "wires": [3]},
            ]
        )
        self.q_layer = tq.RandomLayer(n_ops=8, wires=list(range(self.n_wires)))
        self.measure = tq.MeasureAll(tq.PauliZ)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        device = x.device
        qdev = tq.QuantumDevice(self.n_wires, bsz=bsz, device=device)
        x = x.view(bsz, 28, 28)
        patches = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                data = torch.stack(
                    [
                        x[:, r, c],
                        x[:, r, c + 1],
                        x[:, r + 1, c],
                        x[:, r + 1, c + 1],
                    ],
                    dim=1,
                )
                self.encoder(qdev, data)
                self.q_layer(qdev)
                measurement = self.measure(qdev)
                patches.append(measurement.view(bsz, 4))
        return torch.cat(patches, dim=1)

# Hybrid quantum‑classical model ---------------------------------------------

class HybridQuanvolutionFraudClassifier(tq.QuantumModule):
    """
    Quantum‑enabled model that combines a random quantum kernel applied
    to 2×2 image patches with a fraud‑detection style linear head.
    """
    def __init__(
        self,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layer_params: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()

        if fraud_input_params is None:
            fraud_input_params = FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
        if fraud_layer_params is None:
            fraud_layer_params = [
                FraudLayerParameters(
                    bs_theta=0.1,
                    bs_phi=0.2,
                    phases=(0.1, 0.1),
                    squeeze_r=(0.1, 0.1),
                    squeeze_phi=(0.0, 0.0),
                    displacement_r=(0.5, 0.5),
                    displacement_phi=(0.0, 0.0),
                    kerr=(0.0, 0.0),
                )
            ]

        self.fraud_head = build_fraud_detection_head(
            fraud_input_params, fraud_layer_params
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.fraud_head(features)
        return F.log_softmax(logits, dim=-1)

# Legacy wrappers ------------------------------------------------------------

class QuanvolutionFilter(tq.QuantumModule):
    """Backwards‑compatible wrapper around the quantum filter."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.qfilter(x)

class QuanvolutionClassifier(tq.QuantumModule):
    """Backwards‑compatible classifier that uses the quantum filter."""
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuantumQuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = [
    "HybridQuanvolutionFraudClassifier",
    "QuanvolutionFilter",
    "QuanvolutionClassifier",
    "FraudLayerParameters",
    "build_fraud_detection_head",
]
