"""Unified hybrid architecture combining QCNN, fraud‑detection and quanvolution concepts.

The design follows a *combination* scaling paradigm:
* Classical neural network layers (fully‑connected, Conv2d) are implemented with PyTorch.
* Quantum circuits are used as feature‑map kernels or quantum‑kernel kernels.
* The final classifier head is shared and trainable across both regimes.

The model is intentionally modular: each sub‑module can be swapped or extended independently.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# --- QCNN sub‑module (classical emulation of the quantum convolution) ---
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

# --- Fraud‑detection sub‑module (classical analogue of the photonic circuit) ---
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
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# --- Quanvolution sub‑module (classical convolutional filter inspired by the quanvolution example) ---
class QuanvolutionFilter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.conv(x)
        return features.view(x.size(0), -1)

class QuanvolutionClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qfilter = QuanvolutionFilter()
        self.linear = nn.Linear(4 * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        features = self.qfilter(x)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

# --- Unified hybrid architecture ---
class UnifiedQCNNFraudQuanvolution(nn.Module):
    """Combines QCNN, fraud‑detection and quanvolution sub‑modules.

    The forward method accepts a dictionary with keys:
        - 'qcnn_input': Tensor of shape (batch, 8)
        - 'fraud_input': Tensor of shape (batch, 2)
        - 'quanv_input': Tensor of shape (batch, 1, 28, 28)
    Returns a dictionary of outputs from each head.
    """
    def __init__(self) -> None:
        super().__init__()
        self.qcnn = QCNNModel()
        # Example fraud parameters; in practice these would be learned.
        dummy_params = FraudLayerParameters(
            bs_theta=0.1, bs_phi=0.2,
            phases=(0.3, 0.4),
            squeeze_r=(0.5, 0.6),
            squeeze_phi=(0.7, 0.8),
            displacement_r=(0.9, 1.0),
            displacement_phi=(1.1, 1.2),
            kerr=(1.3, 1.4)
        )
        self.fraud = build_fraud_detection_program(dummy_params, [])
        self.quanv = QuanvolutionClassifier()

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        if "qcnn_input" in inputs:
            outputs["qcnn_output"] = self.qcnn(inputs["qcnn_input"])
        if "fraud_input" in inputs:
            outputs["fraud_output"] = self.fraud(inputs["fraud_input"])
        if "quanv_input" in inputs:
            outputs["quanv_output"] = self.quanv(inputs["quanv_input"])
        return outputs

__all__ = ["UnifiedQCNNFraudQuanvolution", "QCNNModel", "FraudLayerParameters",
           "build_fraud_detection_program", "QuanvolutionFilter", "QuanvolutionClassifier"]
