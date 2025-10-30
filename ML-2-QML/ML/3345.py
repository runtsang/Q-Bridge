import torch
from torch import nn
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
        dtype=torch.float32)
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
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

class RBFKernelLayer(nn.Module):
    """Compute RBF kernel between input features and a fixed set of support vectors."""
    def __init__(self, support_vectors: torch.Tensor, gamma: float = 1.0):
        super().__init__()
        self.support_vectors = support_vectors
        self.gamma = gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        diff = x.unsqueeze(1) - self.support_vectors.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=2)
        return torch.exp(-self.gamma * dist_sq)

class FraudDetectionHybrid(nn.Module):
    """Hybrid classical model combining photonic-inspired layers and an RBF kernel."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layers: Iterable[FraudLayerParameters],
                 support_vectors: torch.Tensor,
                 gamma: float = 1.0) -> None:
        super().__init__()
        modules = [_layer_from_params(input_params, clip=False)]
        modules.extend(_layer_from_params(l, clip=True) for l in layers)
        self.feature_extractor = nn.Sequential(*modules)
        self.kernel_layer = RBFKernelLayer(support_vectors, gamma)
        self.classifier = nn.Linear(support_vectors.shape[0], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        kernel_features = self.kernel_layer(features)
        logits = self.classifier(kernel_features)
        return logits

def build_fraud_detection_hybrid(input_params: FraudLayerParameters,
                                 layers: Iterable[FraudLayerParameters],
                                 support_vectors: torch.Tensor,
                                 gamma: float = 1.0) -> FraudDetectionHybrid:
    return FraudDetectionHybrid(input_params, layers, support_vectors, gamma)

__all__ = ["FraudLayerParameters", "FraudDetectionHybrid", "build_fraud_detection_hybrid"]
