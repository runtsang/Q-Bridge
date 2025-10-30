import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence

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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class QuanvolutionFilter(nn.Module):
    """Classical convolutional filter inspired by the quanvolution example."""
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QCNNModel(nn.Module):
    """Stack of fully connected layers emulating the quantum convolution steps."""
    def __init__(self) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class FraudDetectionHybridModel(nn.Module):
    """
    Hybrid fraud detection model that combines a quantum-inspired feature extractor
    (QuanvolutionFilter), a classical QCNN backbone, and a photonic-inspired linear head.
    """
    def __init__(self, fraud_params: FraudLayerParameters) -> None:
        super().__init__()
        self.quantum_filter = QuanvolutionFilter()
        self.qcnn = QCNNModel()
        self.linear = nn.Linear(4, 1)
        self._prepare_photonic_params(fraud_params)

    def _prepare_photonic_params(self, p: FraudLayerParameters) -> None:
        self.bs_theta = p.bs_theta
        self.bs_phi = p.bs_phi
        self.phases = torch.tensor(p.phases, dtype=torch.float32)
        self.squeeze_r = torch.tensor(p.squeeze_r, dtype=torch.float32)
        self.squeeze_phi = torch.tensor(p.squeeze_phi, dtype=torch.float32)
        self.displacement_r = torch.tensor(p.displacement_r, dtype=torch.float32)
        self.displacement_phi = torch.tensor(p.displacement_phi, dtype=torch.float32)
        self.kerr = torch.tensor(p.kerr, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qfeat = self.quantum_filter(x)          # (batch, 4*14*14)
        qfeat = qfeat.view(-1, 4 * 14 * 14)
        qcnn_out = self.qcnn(qfeat)
        out = self.linear(qcnn_out)
        return torch.sigmoid(out)

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model that mirrors the layered structure."""
    modules = [FraudDetectionHybridModel(input_params)]
    modules.extend(FraudDetectionHybridModel(layer) for layer in layers)
    modules.append(nn.Linear(1, 1))
    return nn.Sequential(*modules)

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybridModel",
    "QuanvolutionFilter",
    "QCNNModel",
    "build_fraud_detection_program",
]
