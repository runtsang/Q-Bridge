"""Hybrid fraud‑detection model that fuses a classical convolutional front‑end with photonic‑inspired fully‑connected layers.

The design mirrors the original `FraudDetection.py` and `Conv.py` seeds while providing a single
callable class that can be used in a pure‑Python training loop or exported to ONNX/TorchScript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import torch
from torch import nn


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


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


class ConvFilter(nn.Module):
    """Return a callable object that emulates the quantum filter with PyTorch ops."""

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data) -> float:
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class FraudDetectionHybrid(nn.Module):
    """
    Hybrid fraud‑detection model that combines a convolutional front‑end with
    photonic‑inspired fully‑connected layers.

    Parameters
    ----------
    conv_kernel_size : int
        Size of the convolutional kernel.
    conv_threshold : float
        Threshold used in the convolutional activation.
    fraud_params : list[FraudLayerParameters]
        List of layer parameters for the photonic part. The first element
        is used as the input layer; the rest are hidden layers.
    """

    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        fraud_params: List[FraudLayerParameters] | None = None,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(conv_kernel_size, conv_threshold)
        self.fraud_params = fraud_params or []

        if self.fraud_params:
            self.model = build_fraud_detection_program(
                self.fraud_params[0], self.fraud_params[1:]
            )
        else:
            self.model = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        inputs : torch.Tensor
            Expected shape (batch, 1, H, W).  The convolutional layer
            operates on each sample independently.

        Returns
        -------
        torch.Tensor
            Fraud‑risk scores.
        """
        # Flatten per sample for the convolutional filter
        batch_size = inputs.shape[0]
        conv_out = torch.stack(
            [torch.tensor(self.conv.run(sample.squeeze(0))) for sample in inputs]
        )
        # Reshape to match the photonic layer input shape (batch, 2)
        conv_out = conv_out.unsqueeze(1).repeat(1, 2)
        return self.model(conv_out)


__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "FraudDetectionHybrid"]
