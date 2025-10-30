"""Hybrid classical model combining convolution, fraud‑detection layers and an RBF kernel."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np


# ------------------------------------------------------------------  Conv  ------------------------------------
class ConvFilter(nn.Module):
    """2‑D convolution filter with a sigmoid activation and a threshold."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        logits = self.conv(data)
        return torch.sigmoid(logits - self.threshold)


# ------------------------------------------------------------------  Fraud  ------------------------------------
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class FraudLayerParameters:
    """Container mirroring the photonic parameters."""
    def __init__(self, bs_theta, bs_phi, phases, squeeze_r, squeeze_phi,
                 displacement_r, displacement_phi, kerr):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr


def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            return outputs * self.scale + self.shift
    return Layer()


def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: list[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)


# ------------------------------------------------------------------  Kernel  ------------------------------------
class Kernel(nn.Module):
    """Radial‑basis function kernel implemented as a torch module."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))


def kernel_matrix(a: list[torch.Tensor], b: list[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


# ------------------------------------------------------------------  Hybrid Class  ------------------------------------
class HybridConvFraudKernel(nn.Module):
    """
    A drop‑in replacement for Conv.py that chains:
    1. 2‑D convolution filter
    2. fraud‑detection style network
    3. optional kernel evaluation between two feature vectors
    """
    def __init__(
        self,
        conv_kernel_size: int = 2,
        conv_threshold: float = 0.0,
        fraud_input_params: FraudLayerParameters | None = None,
        fraud_layers: list[FraudLayerParameters] | None = None,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel_size, threshold=conv_threshold)
        if fraud_input_params is None:
            fraud_input_params = FraudLayerParameters(
                bs_theta=0.0, bs_phi=0.0, phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0), squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0), displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0))
        if fraud_layers is None:
            fraud_layers = []
        self.fraud_net = build_fraud_detection_program(fraud_input_params, fraud_layers)
        self.kernel = Kernel(gamma=kernel_gamma)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image : torch.Tensor
            Shape (1, 1, H, W) – grayscale image with a single channel.
        Returns
        -------
        torch.Tensor
            Output of the fraud‑detection network as a scalar.
        """
        conv_out = self.conv(image)
        # Flatten to (batch, 2) – assume conv_out shape (1,1,2,2)
        flat = conv_out.view(conv_out.size(0), -1)[:,:2]
        return self.fraud_net(flat)

    def compute_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the kernel value between two feature vectors."""
        return self.kernel(x, y)
