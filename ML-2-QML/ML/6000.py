import torch
from torch import nn
from dataclasses import dataclass
from typing import Iterable, Sequence

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
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class QCNNFraudHybrid(nn.Module):
    """
    Hybrid classical network that merges a QCNN-style feature extractor with a fraud-detection style
    fully‑connected stack.  The model processes two independent inputs: a high‑dimensional feature
    vector for the QCNN branch and a two‑dimensional vector for the fraud branch.  The outputs can
    be concatenated or combined for downstream tasks.
    """

    def __init__(
        self,
        qcnn_input_dim: int = 8,
        qcnn_hidden_dim: int = 16,
        fraud_input_dim: int = 2,
        fraud_hidden_dim: int = 4,
        fraud_layers: int = 2,
    ) -> None:
        super().__init__()
        # QCNN branch
        self.qcnn_branch = nn.Sequential(
            nn.Linear(qcnn_input_dim, qcnn_hidden_dim), nn.Tanh(),
            nn.Linear(qcnn_hidden_dim, qcnn_hidden_dim), nn.Tanh(),
            nn.Linear(qcnn_hidden_dim, qcnn_hidden_dim), nn.Tanh(),
            nn.Linear(qcnn_hidden_dim, qcnn_hidden_dim // 2), nn.Tanh(),
            nn.Linear(qcnn_hidden_dim // 2, qcnn_hidden_dim // 4), nn.Tanh(),
            nn.Linear(qcnn_hidden_dim // 4, qcnn_hidden_dim // 8), nn.Tanh(),
            nn.Linear(qcnn_hidden_dim // 8, 1),
        )
        # Fraud branch
        dummy_params = FraudLayerParameters(
            bs_theta=0.0,
            bs_phi=0.0,
            phases=(0.0, 0.0),
            squeeze_r=(0.0, 0.0),
            squeeze_phi=(0.0, 0.0),
            displacement_r=(0.0, 0.0),
            displacement_phi=(0.0, 0.0),
            kerr=(0.0, 0.0),
        )
        self.fraud_branch = build_fraud_detection_program(
            input_params=dummy_params,
            layers=[dummy_params] * fraud_layers,
        )

    def forward(
        self,
        qcnn_input: torch.Tensor,
        fraud_input: torch.Tensor,
    ) -> torch.Tensor:  # type: ignore[override]
        qcnn_out = self.qcnn_branch(qcnn_input)
        fraud_out = self.fraud_branch(fraud_input)
        return torch.cat([qcnn_out, fraud_out], dim=-1)

__all__ = ["FraudLayerParameters", "build_fraud_detection_program", "QCNNFraudHybrid"]
