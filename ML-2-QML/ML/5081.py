import torch
from torch import nn
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class FraudLayerParameters:
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

class ConvLayer(nn.Module):
    def __init__(self, kernel_size: int = 2, threshold: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.conv(x)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean()

class FullyConnectedLayer(nn.Module):
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, thetas: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.linear(thetas)).mean()

class QCNNModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class HybridGen445(nn.Module):
    def __init__(self, kernel_size: int = 2, n_features: int = 1, clip_params: bool = True) -> None:
        super().__init__()
        self.conv = ConvLayer(kernel_size=kernel_size, threshold=0.0)
        self.fcl = FullyConnectedLayer(n_features=n_features)
        self.qcnn = QCNNModel()
        self.clip_params = clip_params
        self.fraud_params = self._random_fraud_params()

    def _random_fraud_params(self) -> FraudLayerParameters:
        return FraudLayerParameters(
            bs_theta=np.random.uniform(-np.pi, np.pi),
            bs_phi=np.random.uniform(-np.pi, np.pi),
            phases=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
            squeeze_r=(np.random.uniform(-3, 3), np.random.uniform(-3, 3)),
            squeeze_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
            displacement_r=(np.random.uniform(-3, 3), np.random.uniform(-3, 3)),
            displacement_phi=(np.random.uniform(-np.pi, np.pi), np.random.uniform(-np.pi, np.pi)),
            kerr=(np.random.uniform(-1, 1), np.random.uniform(-1, 1)),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(data).unsqueeze(0)
        fcl_out = self.fcl(conv_out)
        qcnn_out = self.qcnn(fcl_out)
        return qcnn_out

    def clip_all(self) -> None:
        if self.clip_params:
            for m in self.modules():
                if hasattr(m, "weight"):
                    m.weight.data.clamp_(-5.0, 5.0)
                if hasattr(m, "bias"):
                    m.bias.data.clamp_(-5.0, 5.0)

__all__ = ["HybridGen445", "FraudLayerParameters"]
