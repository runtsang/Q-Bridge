from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Dict
import numpy as np
import torch
from torch import nn

# ------------------------------------------------------------------
# Classical fraud‑detection building blocks
# ------------------------------------------------------------------
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
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

# ------------------------------------------------------------------
# Classical self‑attention
# ------------------------------------------------------------------
class ClassicalSelfAttention:
    """
    Mimics the quantum self‑attention block but operates on NumPy arrays.
    """
    def __init__(self, embed_dim: int):
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1),
            dtype=torch.float32,
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()

# ------------------------------------------------------------------
# Fully‑connected layer helper
# ------------------------------------------------------------------
class FullyConnectedLayer(nn.Module):
    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        values = torch.as_tensor(list(thetas), dtype=torch.float32).view(-1, 1)
        expectation = torch.tanh(self.linear(values)).mean(dim=0)
        return expectation.detach().numpy()

def FCL() -> FullyConnectedLayer:
    return FullyConnectedLayer()

# ------------------------------------------------------------------
# Classical classifier factory
# ------------------------------------------------------------------
def build_classifier_circuit(
    num_features: int, depth: int
) -> Tuple[nn.Module, Sequence[int], Sequence[int], Sequence[int]]:
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

# ------------------------------------------------------------------
# Main hybrid class
# ------------------------------------------------------------------
class FraudDetectionHybrid(nn.Module):
    """
    End‑to‑end PyTorch module that chains fraud‑detection layers,
    self‑attention, a fully‑connected layer and a classifier.
    """
    def __init__(
        self,
        fraud_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        attention_params: Dict[str, np.ndarray],
        classifier_params: Dict[str, int],
    ) -> None:
        super().__init__()
        self.fraud_net = build_fraud_detection_program(fraud_params, fraud_layers)
        self.attention = ClassicalSelfAttention(attention_params["embed_dim"])
        self.classifier, self.enc, self.wsize, self.obs = build_classifier_circuit(
            classifier_params["num_features"], classifier_params["depth"]
        )
        self.fcl = FCL()
        self.attention_params = attention_params

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        fraud_out = self.fraud_net(inputs)
        attn_out = self.attention.run(
            rotation_params=self.attention_params["rotation"],
            entangle_params=self.attention_params["entangle"],
            inputs=inputs.numpy(),
        )
        fcl_out = self.fcl.run([1.0])  # placeholder theta
        # Concatenate all intermediate signals
        combined = torch.cat(
            [fraud_out, torch.tensor(attn_out), torch.tensor(fcl_out)], dim=1
        )
        logits = self.classifier(combined)
        return logits

__all__ = [
    "FraudDetectionHybrid",
    "build_fraud_detection_program",
    "ClassicalSelfAttention",
    "FCL",
    "build_classifier_circuit",
]
