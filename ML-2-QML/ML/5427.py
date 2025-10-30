import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple

@dataclass
class FraudLayerParameters:
    """Parameters describing a fully connected layer in the classical model."""
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
        [[params.bs_theta, params.bs_phi],
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
        def __init__(self):
            super().__init__()
            self.linear = linear
            self.activation = activation
            self.register_buffer("scale", scale)
            self.register_buffer("shift", shift)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            out = self.activation(self.linear(inputs))
            out = out * self.scale + self.shift
            return out

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

class FraudLSTMEncoder(nn.Module):
    """Classical LSTM encoder for transaction sequences."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hn, _) = self.lstm(x)
        return hn.squeeze(0)

class QuantumKernelApprox(nn.Module):
    """Approximate quantum kernel using classical RBF."""
    def __init__(self, gamma: float = 1.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        diff = x - y
        return torch.exp(-self.gamma * torch.sum(diff * diff, dim=-1, keepdim=True))

class FraudDetectionHybrid(nn.Module):
    """Hybrid fraud detection model combining classical photonic layers,
    LSTM encoder, kernel similarity and a sigmoid head."""
    def __init__(self,
                 input_params: FraudLayerParameters,
                 layer_params: Sequence[FraudLayerParameters],
                 lstm_hidden: int,
                 gamma: float = 1.0):
        super().__init__()
        self.photo = build_fraud_detection_program(input_params, layer_params)
        self.encoder = FraudLSTMEncoder(input_dim=2, hidden_dim=lstm_hidden)
        self.kernel = QuantumKernelApprox(gamma)
        self.classifier = nn.Linear(lstm_hidden, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq shape: (batch, seq_len, 2)
        batch, seq_len, _ = seq.shape
        # photonic feature extraction
        features = self.photo(seq.reshape(-1, 2)).reshape(batch, seq_len, -1)
        # LSTM encoding
        encoded = self.encoder(features)
        # kernel similarity with a learned prototype
        proto = self.register_parameter("proto", nn.Parameter(torch.randn_like(encoded)))
        similarity = self.kernel(encoded, proto)
        logits = self.classifier(similarity)
        probs = torch.sigmoid(logits)
        return torch.cat([probs, 1 - probs], dim=-1)

__all__ = ["FraudLayerParameters",
           "build_fraud_detection_program",
           "FraudLSTMEncoder",
           "QuantumKernelApprox",
           "FraudDetectionHybrid"]
