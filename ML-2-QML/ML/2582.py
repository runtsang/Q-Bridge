import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

# Attempt to import the quantum kernel implementation; fall back to a classical RBF if unavailable.
try:
    from.quantum_module import FraudDetectionHybrid as QuantumFraudDetectionHybrid
except Exception:  # pragma: no cover
    class QuantumFraudDetectionHybrid(nn.Module):
        """Fallback quantum kernel using a classical RBF."""
        def __init__(self, *_, **__):
            super().__init__()
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            diff = x - y
            return torch.exp(-torch.sum(diff * diff, dim=-1, keepdim=True))

@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic‑inspired layer."""
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
    # Build a trainable 2‑to‑2 linear layer with custom bias and scaling.
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
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

    return Layer()

def build_fraud_detection_backbone(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud‑detection network combining a photonic‑style backbone and a quantum kernel."""
    def __init__(self, n_layers: int = 3, kernel_gamma: float = 1.0, ref_size: int = 8):
        super().__init__()
        # Classical backbone
        input_params = FraudLayerParameters(
            bs_theta=0.5, bs_phi=0.3,
            phases=(0.1, -0.1),
            squeeze_r=(0.2, 0.2),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.1, 0.1),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        layer_params = [
            FraudLayerParameters(
                bs_theta=0.4, bs_phi=0.2,
                phases=(0.05, -0.05),
                squeeze_r=(0.15, 0.15),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.05, 0.05),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            for _ in range(n_layers)
        ]
        self.backbone = build_fraud_detection_backbone(input_params, layer_params)

        # Quantum kernel
        self.kernel = QuantumFraudDetectionHybrid()
        self.kernel_gamma = kernel_gamma

        # Reference feature set for kernel evaluation
        self.register_buffer(
            "reference_features",
            torch.randn(ref_size, 2)
        )
        self.classifier = nn.Linear(ref_size, 1)

    def _kernel_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute a batch kernel matrix using the quantum kernel."""
        batch = x.shape[0]
        ref = y.shape[0]
        out = torch.empty(batch, ref, device=x.device, dtype=torch.float32)
        for i in range(batch):
            for j in range(ref):
                out[i, j] = self.kernel(x[i], y[j])
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning a fraud‑score per sample."""
        features = self.backbone(x)  # shape: (batch, 1)
        # Expand to match kernel input dimensionality
        features = features.squeeze(-1)
        kernel_vals = self._kernel_matrix(features, self.reference_features)
        logits = self.classifier(kernel_vals)
        return logits
