import torch
from torch import nn
import numpy as np
from dataclasses import dataclass

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

def build_fraud_feature_extractor(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 2))
    return nn.Sequential(*modules)

class ClassicalSelfAttention(nn.Module):
    """Simple self‑attention block operating on feature tensors."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj   = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        query = self.query_proj(x)
        key   = self.key_proj(x)
        value = self.value_proj(x)
        scores = torch.softmax(query @ key.transpose(-2, -1) / np.sqrt(self.embed_dim), dim=-1)
        return scores @ value

class QCNNModel(nn.Module):
    """Classical imitation of the QCNN convolution‑pooling stack."""
    def __init__(self, input_dim: int = 8) -> None:
        super().__init__()
        self.feature_map = nn.Sequential(nn.Linear(input_dim, 16), nn.Tanh())
        self.conv1 = nn.Sequential(nn.Linear(16, 16), nn.Tanh())
        self.pool1 = nn.Sequential(nn.Linear(16, 12), nn.Tanh())
        self.conv2 = nn.Sequential(nn.Linear(12, 8), nn.Tanh())
        self.pool2 = nn.Sequential(nn.Linear(8, 4), nn.Tanh())
        self.conv3 = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
        self.head  = nn.Linear(4, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.feature_map(inputs)
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        return torch.sigmoid(self.head(x))

class QCNNHybridModel(nn.Module):
    """Hybrid model combining fraud detection, self‑attention, and QCNN."""
    def __init__(
        self,
        fraud_input_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        attention_dim: int = 4,
        qcnn_input_dim: int = 8,
    ) -> None:
        super().__init__()
        self.fraud_extractor = build_fraud_feature_extractor(fraud_input_params, fraud_layers)
        self.attention       = ClassicalSelfAttention(attention_dim)
        self.qcnn            = QCNNModel(input_dim=qcnn_input_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.fraud_extractor(inputs)
        x = self.attention(x)
        # Flatten if needed for QCNN
        x = x.view(x.size(0), -1)
        return self.qcnn(x)

def QCNNHybrid() -> QCNNHybridModel:
    """Factory returning a fully configured hybrid QCNN model."""
    # Example hyper‑parameters (should be tuned per task)
    input_params = FraudLayerParameters(
        bs_theta=0.5, bs_phi=0.3,
        phases=(0.1, 0.2),
        squeeze_r=(0.4, 0.4),
        squeeze_phi=(0.0, 0.0),
        displacement_r=(0.1, 0.1),
        displacement_phi=(0.0, 0.0),
        kerr=(0.0, 0.0),
    )
    layers = [
        FraudLayerParameters(
            bs_theta=0.6, bs_phi=0.4,
            phases=(0.2, 0.3),
            squeeze_r=(0.5, 0.5),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.2, 0.2),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
    ]
    return QCNNHybridModel(
        fraud_input_params=input_params,
        fraud_layers=layers,
        attention_dim=4,
        qcnn_input_dim=8,
    )

__all__ = ["QCNNHybridModel", "QCNNHybrid", "FraudLayerParameters", "build_fraud_feature_extractor", "ClassicalSelfAttention"]
