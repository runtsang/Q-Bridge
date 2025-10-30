from __future__ import annotations

import torch
from torch import nn
import torchquantum as tq
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    """Photonic‑style layer parameters used in the classical branch."""
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

class FraudDetectionHybridModel(nn.Module):
    """Hybrid fraud‑detection model: photonic layers + quantum feature extractor."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        num_qubits: int = 4,
    ) -> None:
        super().__init__()
        # Classical photonic stack
        self.classical = nn.Sequential(
            _layer_from_params(input_params, clip=False),
            *(_layer_from_params(p, clip=True) for p in layer_params),
            nn.Linear(2, 1),
        )
        # Quantum part
        self.num_qubits = num_qubits
        self.encoder = SimpleEncoder(num_qubits)
        self.quantum_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.head = nn.Linear(num_qubits, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Classical output
        cls_out = self.classical(x)
        # Quantum feature extraction
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_qubits, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.quantum_layer(qdev)
        features = self.measure(qdev)
        q_out = self.head(features)
        return cls_out + q_out

class SimpleEncoder(tq.QuantumModule):
    """Encode a 2‑dimensional feature vector into rotation angles."""
    def __init__(self, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        for i in range(self.num_qubits):
            angle = x[:, i % x.shape[1]]
            tq.RY(angle, wires=i)(qdev)

def generate_superposition_data(num_features: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic data for regression‑style experiments."""
    x = torch.rand(samples, num_features) * 2 - 1
    angles = x.sum(dim=1)
    y = torch.sin(angles) + 0.1 * torch.cos(2 * angles)
    return x, y

class FraudDataset(torch.utils.data.Dataset):
    """Dataset yielding 2‑dimensional features and a scalar target."""
    def __init__(self, samples: int, num_features: int = 2) -> None:
        self.x, self.y = generate_superposition_data(num_features, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return self.x.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {"features": self.x[idx], "target": self.y[idx]}

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionHybridModel",
    "FraudDataset",
    "generate_superposition_data",
]
