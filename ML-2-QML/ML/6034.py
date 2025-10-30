import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    """Parameters for a single fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class ConvFilter(nn.Module):
    """
    Classical 2‑D convolutional filter that emulates a quanvolution.
    Uses a single‑channel Conv2d followed by a sigmoid and mean reduction.
    """
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0,
                 bias: bool = True, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size,
                              bias=bias, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W] or [B, H, W, 1] – batch of image patches.
        Returns:
            [B] – mean sigmoid‑activated convolution output.
        """
        if x.ndim == 4:  # [B, H, W, 1]
            x = x.permute(0, 3, 1, 2)
        logits = self.conv(x)
        act = torch.sigmoid(logits - self.threshold)
        return act.mean(dim=(1, 2, 3)).squeeze()

class FraudLayer(nn.Module):
    """
    Classical analogue of a photonic fraud‑detection layer.
    Implements a 2‑D linear → tanh → scale‑shift transformation.
    """
    def __init__(self, params: FraudLayerParameters, clip: bool = False) -> None:
        super().__init__()
        weight = torch.tensor([[params.bs_theta, params.bs_phi],
                               [params.squeeze_r[0], params.squeeze_r[1]]],
                              dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer('scale',
                             torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer('shift',
                             torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        return out * self.scale + self.shift

def build_fraud_detection_network(params_list: Sequence[FraudLayerParameters]) -> nn.Sequential:
    """
    Build an nn.Sequential that mirrors the photonic fraud‑detection stack.
    The first layer is un‑clipped; subsequent layers are clipped to avoid
    exploding weights during training.
    """
    modules = []
    if not params_list:
        raise ValueError("At least one FraudLayerParameters instance is required.")
    modules.append(FraudLayer(params_list[0], clip=False))
    for p in params_list[1:]:
        modules.append(FraudLayer(p, clip=True))
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class ConvFraudHybrid(nn.Module):
    """
    Classical hybrid model that combines a ConvFilter with a fraud‑detection
    network.  The model can be trained end‑to‑end with any PyTorch optimizer.
    """
    def __init__(self,
                 conv_kernel_size: int = 2,
                 conv_threshold: float = 0.0,
                 fraud_params: Sequence[FraudLayerParameters] = None) -> None:
        super().__init__()
        self.conv = ConvFilter(kernel_size=conv_kernel_size,
                               threshold=conv_threshold)
        if fraud_params is None:
            fraud_params = []
        self.fraud_net = build_fraud_detection_network(fraud_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ConvFilter produces a scalar per patch; duplicate for 2‑dim input.
        conv_out = self.conv(x).unsqueeze(-1)          # [B, 1]
        inp = conv_out.repeat(1, 2)                    # [B, 2]
        return self.fraud_net(inp)

__all__ = ["FraudLayerParameters", "ConvFilter",
           "FraudLayer", "build_fraud_detection_network",
           "ConvFraudHybrid"]
