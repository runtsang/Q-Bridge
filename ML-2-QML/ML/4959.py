import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

# ────────────────────── Photonic fraud‑detection layer  ──────────────────────
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            outputs = self.activation(self.linear(inputs))
            outputs = outputs * self.scale + self.shift
            return outputs

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ────────────────────── Classical helpers  ──────────────────────
# convolution filter
def Conv():
    class ConvFilter(nn.Module):
        def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
            super().__init__()
            self.kernel_size = kernel_size
            self.threshold = threshold
            self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

        def run(self, data) -> torch.Tensor:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
            logits = self.conv(tensor)
            activations = torch.sigmoid(logits - self.threshold)
            # broadcast to 8‑dim vector to feed downstream components
            return activations.mean().repeat(8)

    return ConvFilter()

# QCNN‑style fully‑connected stack
def QCNN():
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

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            x = self.feature_map(inputs)
            x = self.conv1(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            x = self.conv3(x)
            return torch.sigmoid(self.head(x))

    return QCNNModel()

# self‑attention
def SelfAttention():
    class ClassicalSelfAttention:
        def __init__(self, embed_dim: int):
            self.embed_dim = embed_dim

        def run(
            self,
            rotation_params: torch.Tensor,
            entangle_params: torch.Tensor,
            inputs: torch.Tensor,
        ) -> torch.Tensor:
            query = inputs @ rotation_params.reshape(self.embed_dim, -1)
            key = inputs @ entangle_params.reshape(self.embed_dim, -1)
            scores = torch.softmax(query @ key.T / torch.sqrt(torch.tensor(self.embed_dim, dtype=torch.float32)), dim=-1)
            return scores @ inputs

    return ClassicalSelfAttention(embed_dim=4)

# ────────────────────── Hybrid wrapper  ──────────────────────
class FraudDetectionHybrid(nn.Module):
    """Combines convolution, QCNN, self‑attention and the photonic fraud‑detection stack."""

    def __init__(
        self,
        fraud_input: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        *,
        conv_kernel: int = 2,
        attention_embed: int = 4,
    ) -> None:
        super().__init__()
        self.conv = Conv()
        self.qcnn = QCNN()
        self.attention = SelfAttention()
        self.fraud_layers = build_fraud_detection_program(fraud_input, fraud_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (batch, 8)  – 8‑dim feature vector
        conv_out = self.conv.run(x)          # (batch, 8)
        qcnn_out = self.qcnn(conv_out)       # (batch, 1)
        # For attention we generate dummy parameters; in practice these would be learned
        rot = torch.randn(4, 4)
        ent = torch.randn(4, 4)
        attention_out = self.attention.run(rot, ent, qcnn_out)  # (batch, 1)
        return self.fraud_layers(attention_out)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "Conv", "QCNN", "SelfAttention", "FraudDetectionHybrid"]
