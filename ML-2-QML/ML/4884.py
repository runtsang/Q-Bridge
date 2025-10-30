import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple, Sequence, Dict
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

class FraudLayer(nn.Module):
    def __init__(self, params: FraudLayerParameters, clip: bool):
        super().__init__()
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
        self.linear = linear
        self.activation = nn.Tanh()
        self.register_buffer("scale", torch.tensor(params.displacement_r, dtype=torch.float32))
        self.register_buffer("shift", torch.tensor(params.displacement_phi, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear(inputs))
        return x * self.scale + self.shift

class FraudDetectionProgram(nn.Sequential):
    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]):
        modules = [FraudLayer(input_params, clip=False)]
        modules += [FraudLayer(layer, clip=True) for layer in layers]
        modules.append(nn.Linear(2, 1))
        super().__init__(*modules)

class SamplerQNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.net(inputs), dim=-1)

class QLSTM(nn.Module):
    """Plain PyTorch LSTM used when quantum resources are unavailable."""
    def __init__(self, input_dim: int, hidden_dim: int, n_qubits: int = 0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor] | None = None):
        return self.lstm(inputs, states)

class UnifiedHybridModel(nn.Module):
    """Hybrid architecture that couples a classical classifier, a fraud‑style block,
    a quantum sampler, and a (potentially) quantum LSTM."""
    def __init__(self, feature_dim: int, hidden_dim: int, seq_len: int, n_qubits: int = 0):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.n_qubits = n_qubits

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.classifier.metadata = {
            "layers": 3,
            "output": 2,
            "activation": "ReLU",
        }

        self.fraud_block = nn.Sequential(
            nn.Linear(2, 2),
            nn.Tanh(),
        )
        self.fraud_block.register_buffer("scale", torch.tensor([1.0, 1.0]))
        self.fraud_block.register_buffer("shift", torch.tensor([0.0, 0.0]))
        self.fraud_block.metadata = {"type": "photonic-style"}

        self.sampler = SamplerQNN()
        self.sampler.metadata = {"sampler": "simple"}

        if n_qubits > 0:
            self.lstm = QLSTM(hidden_dim, hidden_dim, n_qubits)
        else:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm.metadata = {"n_qubits": n_qubits}

        self.parameter_registry: Dict[str, torch.Tensor] = {
            name: param for name, param in self.named_parameters()
        }

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        if "features" in inputs:
            out["classifier"] = self.classifier(inputs["features"])
        if "fraud_input" in inputs:
            out["fraud"] = self.fraud_block(inputs["fraud_input"])
        if "sampler_input" in inputs:
            out["sampler"] = self.sampler(inputs["sampler_input"])
        if "sequence" in inputs:
            seq_out, _ = self.lstm(inputs["sequence"])
            out["lstm"] = seq_out
        return out

def build_classifier_circuit(num_features: int, depth: int) -> Tuple[nn.Module, Iterable[int], Iterable[int], list[int]]:
    layers: list[nn.Module] = []
    in_dim = num_features
    encoding = list(range(num_features))
    weight_sizes: list[int] = []
    for _ in range(depth):
        linear = nn.Linear(in_dim, num_features)
        layers.append(linear)
        layers.append(nn.ReLU())
        weight_sizes.append(linear.weight.numel() + linear.bias.numel())
        in_dim = num_features
    head = nn.Linear(in_dim, 2)
    layers.append(head)
    weight_sizes.append(head.weight.numel() + head.bias.numel())
    network = nn.Sequential(*layers)
    observables = list(range(2))
    return network, encoding, weight_sizes, observables

def build_fraud_detection_program(input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> nn.Sequential:
    return FraudDetectionProgram(input_params, layers)

__all__ = [
    "UnifiedHybridModel",
    "build_classifier_circuit",
    "build_fraud_detection_program",
    "SamplerQNN",
    "QLSTM",
]
