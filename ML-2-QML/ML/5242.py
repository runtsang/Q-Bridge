import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np

# ----------------------------------------------------------------------
# Classical fraud‑detection utilities (borrowed from reference [2])
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            outputs = self.activation(self.linear(inputs))
            return outputs * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ----------------------------------------------------------------------
# Simple estimator‑style regression head (borrowed from reference [3])
# ----------------------------------------------------------------------
class EstimatorQNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.net(inputs)

# ----------------------------------------------------------------------
# RBF kernel utility (borrowed from reference [4])
# ----------------------------------------------------------------------
class KernalAnsatz(nn.Module):
    """Placeholder maintaining compatibility with the quantum interface."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class Kernel(nn.Module):
    """RBF kernel module that wraps :class:`KernalAnsatz`."""
    def __init__(self, gamma: float = 1.0) -> None:
        super().__init__()
        self.ansatz = KernalAnsatz(gamma)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.view(1, -1)
        y = y.view(1, -1)
        return self.ansatz(x, y).squeeze()

def kernel_matrix(a: Sequence[torch.Tensor], b: Sequence[torch.Tensor], gamma: float = 1.0) -> np.ndarray:
    kernel = Kernel(gamma)
    return np.array([[kernel(x, y).item() for y in b] for x in a])

# ----------------------------------------------------------------------
# Main hybrid QCNN model
# ----------------------------------------------------------------------
class QCNNHybrid(nn.Module):
    """
    A hybrid QCNN that fuses:
      * classical convolution‑style FC layers (from reference [1]),
      * fraud‑detection inspired layers (from reference [2]),
      * a regression head (from reference [3]),
      * an RBF kernel utility (from reference [4]).
    The architecture mirrors the quantum circuit while adding a fraud‑detection
    preprocessing block and a kernel module for downstream kernel‑based methods.
    """
    def __init__(
        self,
        fraud_params: Iterable[FraudLayerParameters] | None = None,
        kernel_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        # Classical convolution‑style feature extractor
        self.feature_map = nn.Sequential(nn.Linear(8, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())

        # Optional fraud‑detection block
        if fraud_params:
            # The first element is treated as the “input” layer
            self.fraud_block = build_fraud_detection_program(
                fraud_params[0], fraud_params[1:]
            )
        else:
            self.fraud_block = nn.Identity()

        # Regression head
        self.head = nn.Linear(4, 1)
        self.kernel = Kernel(kernel_gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(x)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.fraud_block(x)
        return torch.sigmoid(self.head(x))

    def kernel_forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the RBF kernel value between two batches."""
        return self.kernel(x, y)

def QCNN() -> QCNNHybrid:
    """Factory returning a fully‑configured hybrid QCNN."""
    return QCNNHybrid()
