from __future__ import annotations

import torch
import torch.nn as nn
import torchquantum as tq
from dataclasses import dataclass
from typing import Iterable, Sequence

@dataclass
class FraudLayerParameters:
    """Photonic‑style layer parameters used in the classical head."""
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

class SimpleEncoder(tq.QuantumModule):
    """Encode a 2‑dimensional feature vector into rotation angles."""
    def __init__(self, num_qubits: int):
        super().__init__()
        self.num_qubits = num_qubits

    def forward(self, qdev: tq.QuantumDevice, x: torch.Tensor) -> None:
        for i in range(self.num_qubits):
            angle = x[:, i % x.shape[1]]
            tq.RY(angle, wires=i)(qdev)

class FraudDetectionQuantumHybrid(tq.QuantumModule):
    """Hybrid quantum‑classical model: quantum encoder + photonic‑style head."""
    def __init__(
        self,
        input_params: FraudLayerParameters,
        layer_params: Iterable[FraudLayerParameters],
        num_qubits: int = 4,
    ) -> None:
        super().__init__()
        self.num_qubits = num_qubits
        self.encoder = SimpleEncoder(num_qubits)
        self.quantum_layer = tq.RandomLayer(n_ops=20, wires=list(range(num_qubits)))
        self.measure = tq.MeasureAll(tq.PauliZ)
        # Classical photonic stack applied to measurement outcomes
        self.classical = nn.Sequential(
            _layer_from_params(input_params, clip=False),
            *(_layer_from_params(p, clip=True) for p in layer_params),
            nn.Linear(2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.num_qubits, bsz=bsz, device=x.device)
        self.encoder(qdev, x)
        self.quantum_layer(qdev)
        features = self.measure(qdev)
        return self.classical(features)

def generate_superposition_data(num_wires: int, samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate superposition states for regression‑style experiments."""
    omega_0 = torch.zeros(2 ** num_wires, dtype=torch.cfloat)
    omega_0[0] = 1.0
    omega_1 = torch.zeros(2 ** num_wires, dtype=torch.cfloat)
    omega_1[-1] = 1.0
    thetas = 2 * torch.pi * torch.rand(samples)
    phis = 2 * torch.pi * torch.rand(samples)
    states = torch.zeros((samples, 2 ** num_wires), dtype=torch.cfloat)
    for i in range(samples):
        states[i] = torch.cos(thetas[i]) * omega_0 + torch.exp(1j * phis[i]) * torch.sin(thetas[i]) * omega_1
    labels = torch.sin(2 * thetas) * torch.cos(phis)
    return states, labels

class RegressionDataset(torch.utils.data.Dataset):
    """Dataset returning quantum states and regression targets."""
    def __init__(self, samples: int, num_wires: int) -> None:
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return self.states.shape[0]

    def __getitem__(self, idx: int):  # type: ignore[override]
        return {
            "states": self.states[idx],
            "target": self.labels[idx],
        }

__all__ = [
    "FraudLayerParameters",
    "FraudDetectionQuantumHybrid",
    "RegressionDataset",
    "generate_superposition_data",
]
