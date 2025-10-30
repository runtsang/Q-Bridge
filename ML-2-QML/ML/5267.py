import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

class QuanvolutionFilter(nn.Module):
    """A 2×2 convolution that expands a single‑channel image to 4 feature maps."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x).view(x.size(0), -1)

class SamplerQNN(nn.Module):
    """A shallow neural sampler that outputs a 2‑class probability vector."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(x), dim=-1)

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

class FraudLayer(nn.Module):
    """Affine transform inspired by the photonic fraud‑detection layer."""
    def __init__(self, params: FraudLayerParameters, clip: bool = False) -> None:
        super().__init__()
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
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(x))
        out = out * self.scale + self.shift
        return out

class FullyConnectedLayer(nn.Module):
    """Simple fully‑connected head that mimics the quantum FCL."""
    def __init__(self, n_features: int = 2) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(x)).mean(dim=0, keepdim=True)

class HybridLayer(nn.Module):
    """Hybrid classical network that chains quanvolution, sampler, fraud‑style, and FC layers."""
    def __init__(self, n_features: int = 2) -> None:
        super().__init__()
        self.quanvolution = QuanvolutionFilter()
        self.reduce = nn.Linear(4 * 14 * 14, 2)
        self.sampler = SamplerQNN()
        # Example fraud‑layer parameters; in practice these would be learned.
        fraud_params = FraudLayerParameters(
            bs_theta=0.5, bs_phi=0.3,
            phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.3, 0.3),
            displacement_phi=(0.0, 0.0),
            kerr=(0.01, 0.01),
        )
        self.fraud = FraudLayer(fraud_params, clip=True)
        self.fc = FullyConnectedLayer(n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quanvolution(x)
        x = self.reduce(x)
        x = self.sampler(x)
        x = self.fraud(x)
        x = self.fc(x)
        return x

__all__ = ["HybridLayer"]
