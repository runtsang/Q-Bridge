from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn
import torch.nn.functional as F
import strawberryfields as sf
from strawberryfields.ops import BSgate, Dgate, Kgate, Rgate, Sgate

@dataclass
class FraudLayerParameters:
    """Parameters for a single photonic fraud‑detection layer."""
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
    """Create a PyTorch module that mimics a photonic layer."""
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
) -> sf.Program:
    """Construct a Strawberry‑Fields program that mirrors the layered photonic design."""
    program = sf.Program(2)
    with program.context as q:
        _apply_layer(q, input_params, clip=False)
        for layer in layers:
            _apply_layer(q, layer, clip=True)
    return program

def _apply_layer(modes: Sequence, params: FraudLayerParameters, *, clip: bool) -> None:
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        Sgate(r if not clip else _clip(r, 5), phi) | modes[i]
    BSgate(params.bs_theta, params.bs_phi) | (modes[0], modes[1])
    for i, phase in enumerate(params.phases):
        Rgate(phase) | modes[i]
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        Dgate(r if not clip else _clip(r, 5), phi) | modes[i]
    for i, k in enumerate(params.kerr):
        Kgate(k if not clip else _clip(k, 1)) | modes[i]

class FraudHybridModel(nn.Module):
    """
    Hybrid architecture that combines:
      * a CNN backbone (mirroring QCNet)
      * a stack of photonic fraud‑detection layers
      * a lightweight quantum‑style head implemented as a linear + sigmoid
    The model can perform binary classification or regression depending on the
    ``regression`` flag.
    """
    def __init__(self,
                 regression: bool = False,
                 num_photonic_layers: int = 3,
                 device: torch.device | str | None = None) -> None:
        super().__init__()
        self.regression = regression
        # Convolutional backbone
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(6, 15, kernel_size=3, stride=2, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(55815, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

        # Photonic layers
        self.photonic_params = [FraudLayerParameters(
            bs_theta=0.1 * i, bs_phi=0.1 * i,
            phases=(0.2, 0.2),
            squeeze_r=(0.1, 0.1),
            squeeze_phi=(0.1, 0.1),
            displacement_r=(0.1, 0.1),
            displacement_phi=(0.1, 0.1),
            kerr=(0.1, 0.1)) for i in range(num_photonic_layers)]
        self.photonic_program = build_fraud_detection_program(
            self.photonic_params[0], self.photonic_params[1:])

        # Quantum‑style head
        self.quantum_head = nn.Linear(self.fc3.out_features, 1)
        self.shift = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.drop1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        if self.regression:
            # regression head
            return self.quantum_head(x).squeeze(-1)
        else:
            # classification head
            logits = self.quantum_head(x).squeeze(-1)
            probs = torch.sigmoid(logits)
            return torch.cat([probs, 1 - probs], dim=-1)

    def get_photonic_program(self) -> sf.Program:
        """Return the underlying Strawberry‑Fields program for inspection."""
        return self.photonic_program

__all__ = ["FraudLayerParameters", "build_fraud_detection_program",
           "FraudHybridModel"]
