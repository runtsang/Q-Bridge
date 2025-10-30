import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable

# ----------------------------------------------------------------------
# Classical fraud‑detection backbone
# ----------------------------------------------------------------------
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
    weight = torch.tensor([[params.bs_theta, params.bs_phi],
                           [params.squeeze_r[0], params.squeeze_r[1]]], dtype=torch.float32)
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
            return outputs * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 2))  # 2‑dimensional feature output
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# Classical RBF kernel embedding
# ----------------------------------------------------------------------
class KernalAnsatz(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

# ----------------------------------------------------------------------
# Combined hybrid model
# ----------------------------------------------------------------------
class CombinedEstimatorQNN(nn.Module):
    """
    Hybrid classical‑quantum inspired regressor.
    Classical branch: fraud‑detection style layers → 2‑dim feature vector.
    Kernel branch: RBF kernel embedding of the feature vector.
    Final regression head maps the concatenated representation to a scalar output.
    """

    def __init__(
        self,
        fraud_input: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        support_vectors: torch.Tensor,
        gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.fraud_net = build_fraud_detection_program(fraud_input, fraud_layers)
        self.kernel = Kernel(gamma)
        # support_vectors shape: (n_support, 2)
        self.support = nn.Parameter(support_vectors.clone().detach())
        # final linear head
        self.regressor = nn.Linear(2 + self.support.shape[0], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 2)
        feat = self.fraud_net(x)  # (batch, 2)
        # compute kernel similarities with support vectors
        kernels = torch.stack([self.kernel(feat, sv) for sv in self.support], dim=1)  # (batch, n_support)
        # concatenate
        combined = torch.cat([feat, kernels], dim=1)
        return self.regressor(combined)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "KernalAnsatz", "Kernel", "CombinedEstimatorQNN"]
