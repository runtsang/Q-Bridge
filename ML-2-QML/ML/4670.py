import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

# Classical convolutional filter (from Conv reference)
class ConvFilter(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        t = data.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(t)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

# Photonic‑inspired layer parameters
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

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y = self.activation(self.linear(x))
            return y * self.scale + self.shift

    return Layer()

class FraudDetectionHybrid(nn.Module):
    def __init__(self, num_layers: int = 3, conv_kernel: int = 2, conv_threshold: float = 0.0) -> None:
        super().__init__()
        self.conv = ConvFilter(conv_kernel, conv_threshold)
        self.layers = nn.ModuleList()

        # First layer without clipping
        first_params = FraudLayerParameters(
            bs_theta=0.5,
            bs_phi=0.3,
            phases=(0.0, 0.0),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(1.0, 1.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0)
        )
        self.layers.append(_layer_from_params(first_params, clip=False))

        # Remaining layers with clipping
        for _ in range(num_layers - 1):
            params = FraudLayerParameters(
                bs_theta=np.random.randn(),
                bs_phi=np.random.randn(),
                phases=(np.random.randn(), np.random.randn()),
                squeeze_r=(np.random.randn(), np.random.randn()),
                squeeze_phi=(np.random.randn(), np.random.randn()),
                displacement_r=(np.random.randn(), np.random.randn()),
                displacement_phi=(np.random.randn(), np.random.randn()),
                kerr=(np.random.randn(), np.random.randn())
            )
            self.layers.append(_layer_from_params(params, clip=True))

        self.final = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        vec = torch.stack([conv_out, conv_out], dim=-1)
        for layer in self.layers:
            vec = layer(vec)
        out = self.final(vec)
        return torch.sigmoid(out)

def generate_fraud_dataset(samples: int = 1000, noise: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic fraud dataset: 2x2 images with binary label.
    """
    X = np.random.randn(samples, 2, 2).astype(np.float32)
    y = (X.sum(axis=(1, 2)) > 0).astype(np.float32)
    X += noise * np.random.randn(*X.shape).astype(np.float32)
    return X, y

__all__ = ["FraudDetectionHybrid", "generate_fraud_dataset"]
