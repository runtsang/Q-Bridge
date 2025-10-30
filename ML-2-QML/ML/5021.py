import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Iterable, Sequence, Optional, List

@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic‑style fraud detection layer."""
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

def _layer_from_params(params: FraudLayerParameters, clip: bool) -> nn.Module:
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
            out = self.activation(self.linear(inputs))
            return out * self.scale + self.shift

    return Layer()

def build_fraud_detection_program(
    input_params: FraudLayerParameters,
    layers: Iterable[FraudLayerParameters],
) -> nn.Sequential:
    modules: List[nn.Module] = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(l, clip=True) for l in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class HybridNATNet(nn.Module):
    """
    Classical hybrid network that blends a compact CNN with optional
    photonic‑style fraud‑detection layers and a flexible head that can
    act as a classifier or regressor.
    """
    def __init__(
        self,
        in_channels: int = 1,
        task: str = "classification",
        fraud_params: Optional[Iterable[FraudLayerParameters]] = None,
    ) -> None:
        super().__init__()
        self.task = task

        # Core convolutional feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
        )
        # Compute flattened size dynamically
        dummy = torch.zeros(1, in_channels, 28, 28)
        flat_size = self.features(dummy).view(1, -1).size(1)

        # Optional fraud‑detection layers
        if fraud_params:
            self.fraud = build_fraud_detection_program(
                next(iter(fraud_params)), fraud_params
            )
            # After fraud layers the output is 1‑dimensional
            self.fc_input = 1
            self.fraud_out_dim = 1
        else:
            self.fraud = None
            self.fc_input = flat_size
            self.fraud_out_dim = 0

        # Final head
        if self.task == "classification":
            self.head = nn.Linear(self.fc_input + self.fraud_out_dim, 2)
            self.activation = nn.Softmax(dim=-1)
        else:
            self.head = nn.Linear(self.fc_input + self.fraud_out_dim, 1)
            self.activation = nn.Identity()

        self.bn = nn.BatchNorm1d(self.head.out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = torch.flatten(x, 1)
        if self.fraud:
            x = self.fraud(x)
        out = self.head(x)
        out = self.bn(out)
        return self.activation(out)

    def set_task(self, task: str) -> None:
        """Switch between 'classification' and'regression'."""
        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or'regression'")
        self.task = task
        # Re‑create head accordingly
        if self.task == "classification":
            self.head = nn.Linear(self.fc_input + self.fraud_out_dim, 2)
            self.activation = nn.Softmax(dim=-1)
        else:
            self.head = nn.Linear(self.fc_input + self.fraud_out_dim, 1)
            self.activation = nn.Identity()

__all__ = ["HybridNATNet", "FraudLayerParameters", "build_fraud_detection_program"]
