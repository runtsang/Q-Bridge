"""Hybrid classical classifier integrating CNN, FC, and fraud‑detection style layers with metadata mirroring the quantum interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch
import torch.nn as nn


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
    """Create a sequential PyTorch model mirroring the fraud‑detection architecture."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class HybridClassifier(nn.Module):
    """Hybrid CNN‑FC‑fraud detection classifier with metadata compatible to the quantum interface."""

    def __init__(
        self,
        num_features: int = 4,
        conv_channels: int = 8,
        conv_out_channels: int = 16,
        fc_units: int = 64,
        fraud_layers: Iterable[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        # Convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(conv_channels, conv_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # Fully connected projector
        self.fc = nn.Sequential(
            nn.Linear(conv_out_channels * 7 * 7, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, num_features),
        )
        self.norm = nn.BatchNorm1d(num_features)

        # Fraud‑detection style layers (optional)
        if fraud_layers is None:
            fraud_layers = []
        self.fraud = build_fraud_detection_program(
            FraudLayerParameters(
                bs_theta=0.0,
                bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            ),
            fraud_layers,
        )

        # Final classifier
        self.classifier = nn.Linear(num_features, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        bsz = x.shape[0]
        feat = self.features(x)
        flat = feat.view(bsz, -1)
        proj = self.fc(flat)
        normed = self.norm(proj)
        fraud_out = self.fraud(normed)
        logits = self.classifier(fraud_out)
        return logits

    def build_classifier_circuit(self) -> Tuple[nn.Module, List[int], List[int], List[int]]:
        """Return the network and metadata mimicking the quantum helper."""
        network = self
        encoding = list(range(self.features[0].in_channels))  # input channels
        weight_sizes = [p.numel() for p in self.parameters()]
        observables = [0, 1]  # class indices
        return network, encoding, weight_sizes, observables


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "HybridClassifier"]
