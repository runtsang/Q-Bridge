from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# 1. Shared parameter container
# --------------------------------------------------------------------------- #
class FraudLayerParameters:
    """Container for parameters shared between the classical and photonic layers."""
    def __init__(
        self,
        bs_theta: float,
        bs_phi: float,
        phases: tuple[float, float],
        squeeze_r: tuple[float, float],
        squeeze_phi: tuple[float, float],
        displacement_r: tuple[float, float],
        displacement_phi: tuple[float, float],
        kerr: tuple[float, float],
    ) -> None:
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

# --------------------------------------------------------------------------- #
# 2. Helper functions for building the photonic‑style dense stack
# --------------------------------------------------------------------------- #
def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))

def _layer_from_params(params: FraudLayerParameters, *, clip: bool) -> nn.Module:
    """Create a single layer that mimics the photonic operations."""
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
    layers: list[FraudLayerParameters],
) -> nn.Sequential:
    """Build a sequential PyTorch model that mirrors the photonic layering."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --------------------------------------------------------------------------- #
# 3. Classical convolutional filter (inspired by Conv.py)
# --------------------------------------------------------------------------- #
class ConvFilter(nn.Module):
    """Small 2‑D convolutional filter that outputs a global feature."""
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        # Expect data shape (batch, 1, H, W)
        logits = self.conv(data)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean(dim=[2, 3])  # global average

# --------------------------------------------------------------------------- #
# 4. Hybrid fraud‑detection network
# --------------------------------------------------------------------------- #
class FraudHybridNet(nn.Module):
    """
    Hybrid classical‑quantum fraud‑detection network.

    1. ConvFilter extracts a spatial feature from the input image.
    2. A stack of photonic‑style dense layers processes the feature.
    3. An optional quantum head refines the final prediction.
    """
    def __init__(
        self,
        conv_kernel: int = 2,
        conv_threshold: float = 0.0,
        dense_layers: int = 3,
        clip: bool = True,
    ) -> None:
        super().__init__()
        self.conv = ConvFilter(conv_kernel, conv_threshold)

        # Build a dummy dense program with random parameters for demonstration.
        # In practice these would be learned during training.
        prev_params = FraudLayerParameters(
            bs_theta=0.0,
            bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(1.0, 1.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        layers = []
        for _ in range(dense_layers):
            params = FraudLayerParameters(
                bs_theta=np.random.randn(),
                bs_phi=np.random.randn(),
                phases=(np.random.randn(), np.random.randn()),
                squeeze_r=(np.random.randn(), np.random.randn()),
                squeeze_phi=(np.random.randn(), np.random.randn()),
                displacement_r=(np.random.randn(), np.random.randn()),
                displacement_phi=(np.random.randn(), np.random.randn()),
                kerr=(np.random.randn(), np.random.randn()),
            )
            layers.append(params)
        self.dense_program = build_fraud_detection_program(prev_params, layers)

        # Placeholder for the quantum head; can be set via attach_quantum_head
        self.quantum_head: nn.Module | None = None

    def attach_quantum_head(self, quantum_head: nn.Module) -> None:
        """Attach a quantum expectation head that accepts a 1‑D tensor."""
        self.quantum_head = quantum_head

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs shape (batch, 1, H, W)
        x = self.conv(inputs)          # (batch, 1)
        x = self.dense_program(x)      # (batch, 1)
        if self.quantum_head is not None:
            x = self.quantum_head(x)
        return x

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "ConvFilter",
    "FraudHybridNet",
]
