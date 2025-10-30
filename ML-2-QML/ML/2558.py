import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

@dataclass
class FraudLayerParameters:
    """Parameters for a fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class QuanvolutionFilter(nn.Module):
    """Classical 2×2 patch filter that emulates the quantum quanvolution idea."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return features.view(x.size(0), -1)

class FraudLayer(nn.Module):
    """Custom fully‑connected layer mirroring the photonic fraud‑detection block."""
    def __init__(self, input_dim: int, params: FraudLayerParameters) -> None:
        super().__init__()
        weight = torch.tensor([[params.bs_theta]*input_dim, [params.bs_phi]*input_dim], dtype=torch.float32)
        bias = torch.tensor(params.phases, dtype=torch.float32)
        weight = weight.clamp(-5.0, 5.0)
        bias = bias.clamp(-5.0, 5.0)
        linear = nn.Linear(input_dim, 2)
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        self.linear = linear
        self.activation = nn.Tanh()
        self.scale = nn.Parameter(torch.tensor(params.displacement_r, dtype=torch.float32))
        self.shift = nn.Parameter(torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out

class HybridQuanvolutionFraudNet(nn.Module):
    """Hybrid architecture: classical quanvolution followed by fraud‑detection layers."""
    def __init__(self, fraud_params: List[FraudLayerParameters]) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        input_dim = 4 * 14 * 14
        self.layers = nn.ModuleList()
        for params in fraud_params:
            self.layers.append(FraudLayer(input_dim, params))
            input_dim = 2
        self.final = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.qfilter(x)
        out = features
        for layer in self.layers:
            out = layer(out)
        logits = self.final(out)
        return F.log_softmax(logits, dim=-1)

__all__ = ["FraudLayerParameters", "QuanvolutionFilter", "FraudLayer", "HybridQuanvolutionFraudNet"]
