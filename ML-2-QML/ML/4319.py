"""Hybrid fraud detection model combining classical layers, a quantum‑inspired kernel, and a sampler QNN."""
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    return nn.Sequential(*modules)

class SamplerQNN(nn.Module):
    """Classical surrogate of the quantum sampler QNN."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

def quantum_kernel_matrix(a: torch.Tensor, b: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """Compute a quantum‑inspired RBF kernel via sine/cosine feature map."""
    def phi(x: torch.Tensor) -> torch.Tensor:
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    a_phi = phi(a)
    b_phi = phi(b)
    diff = a_phi.unsqueeze(1) - b_phi.unsqueeze(0)  # shape (len(a), len(b), 2*dim)
    dist_sq = torch.sum(diff * diff, dim=-1)
    return torch.exp(-gamma * dist_sq)

class FraudDetectionHybridModel(nn.Module):
    """Hybrid fraud‑detection model combining classical layers, a quantum kernel, and a sampler QNN."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layer_params: Iterable[FraudLayerParameters],
                 reference_set: torch.Tensor,
                 kernel_gamma: float = 1.0):
        super().__init__()
        self.classical_net = build_fraud_detection_program(input_params, layer_params)
        self.reference_set = reference_set  # shape (n_ref, 2)
        self.kernel_gamma = kernel_gamma
        self.sampler = SamplerQNN()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass producing fraud probability."""
        # Classical feature extraction
        features = self.classical_net(x)  # shape (batch, 2)
        # Compute kernel similarity with reference set
        k = quantum_kernel_matrix(features, self.reference_set, self.kernel_gamma)  # (batch, n_ref)
        # Aggregate similarities (e.g., mean) and feed to sampler
        aggregated = k.mean(dim=1, keepdim=True)  # (batch, 1)
        probs = self.sampler(aggregated)  # (batch, 2)
        return probs

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "SamplerQNN",
    "FraudDetectionHybridModel",
]
