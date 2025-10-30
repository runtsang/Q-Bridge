import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Tuple, List

class QuanvolutionFilter(nn.Module):
    """Classical 2x2 patch embedding using a small MLP."""
    def __init__(self, patch_size: int = 2, out_dim: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.out_dim = out_dim
        self.embedding = nn.Sequential(
            nn.Linear(patch_size * patch_size, 8),
            nn.ReLU(),
            nn.Linear(8, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        assert c == 1, "Only single‑channel input supported."
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(b, -1, self.patch_size * self.patch_size)
        features = self.embedding(patches)
        return features.view(b, -1)

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

class FraudLayer(nn.Module):
    """Custom layer inspired by photonic fraud detection."""
    def __init__(self, params: FraudLayerParameters, clip: bool = False):
        super().__init__()
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
        self.linear = linear
        self.activation = activation
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        out = self.activation(self.linear(inputs))
        return out * self.scale + self.shift

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters]
) -> nn.Sequential:
    """Build a sequential model mirroring the photonic fraud detection."""
    modules = [FraudLayer(input_params, clip=False)]
    modules.extend(FraudLayer(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], List[int]]:
    """Construct a classical feed‑forward classifier mimicking the quantum ansatz."""
    layers = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.extend([linear, nn.ReLU()])
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

class EstimatorQNN(nn.Module):
    """Regression head similar to the EstimatorQNN example."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)

class QuanvolutionClassifier(nn.Module):
    """Hybrid classifier using a quantum‑inspired feature extractor and a classical head."""
    def __init__(self, num_classes: int = 10, depth: int = 3):
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.fraud = FraudLayer(
            FraudLayerParameters(
                bs_theta=0.0, bs_phi=0.0,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(1.0, 1.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0)
            ), clip=False)
        self.classifier, _, _, _ = build_classifier_circuit(4 * 14 * 14, depth)
        self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.qfilter(x)
        feat = self.fraud(feat)
        logits = self.linear(feat)
        return F.log_softmax(logits, dim=-1)

__all__ = [
    "QuanvolutionFilter",
    "FraudLayerParameters",
    "FraudLayer",
    "build_fraud_detection_program",
    "build_classifier_circuit",
    "EstimatorQNN",
    "QuanvolutionClassifier",
]
