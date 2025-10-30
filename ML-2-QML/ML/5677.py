"""Hybrid fraud‑detection and regression model combining photonic‑style classical layers and a variational quantum regression head.

The module defines:
* FraudLayerParameters – same struct as in the original fraud‑detection seed.
* build_fraud_detection_program – helper from the photonic seed.
* FraudDetectionRegressionHybrid – torch.nn.Module that builds a classical backbone and a quantum regression head.
* HybridRegressionDataset – dataset generating superposition data (from the quantum‑regression seed).
"""

import torch
from torch import nn
from typing import Iterable
import numpy as np

# Import the quantum head from the quantum module.
# The quantum module is expected to provide a class named QuantumRegressionHead.
# It is defined in the qml_code block below.
try:
    from.quantum_module import QuantumRegressionHead
except Exception:
    # Fallback: if the quantum module cannot be imported, define a dummy placeholder.
    class QuantumRegressionHead(nn.Module):
        def __init__(self, *_, **__):
            super().__init__()
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, device=x.device, dtype=torch.float32)

class FraudLayerParameters:
    """Parameters describing a fully‑connected‑like 2‑mode photonic‑style layer."""
    def __init__(self, bs_theta: float, bs_phi: float, phases: tuple[float, float],
                 squeeze_r: tuple[float, float], squeeze_phi: tuple[float, float],
                 displacement_r: tuple[float, float], displacement_phi: tuple[float, float],
                 kerr: tuple[float, float]):
        self.bs_theta = bs_theta
        self.bs_phi = bs_phi
        self.phases = phases
        self.squeeze_r = squeeze_r
        self.squeeze_phi = squeeze_phi
        self.displacement_r = displacement_r
        self.displacement_phi = displacement_phi
        self.kerr = kerr

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
    """Create a sequential PyTorch model mirroring the layered structure."""
    modules = [_layer_from_params(input_params, clip=False)]
    modules.extend(_layer_from_params(layer, clip=True) for layer in layers)
    modules.append(nn.Linear(2, 1))
    return nn.Sequential(*modules)

class FraudDetectionRegressionHybrid(nn.Module):
    """Hybrid model: photonic‑style classical backbone + quantum regression head."""
    def __init__(self, input_params: FraudLayerParameters,
                 layer_params: Iterable[FraudLayerParameters],
                 num_wires: int = 4):
        super().__init__()
        self.classical = build_fraud_detection_program(input_params, layer_params)
        self.quantum_head = QuantumRegressionHead(num_wires)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the classical backbone followed by the quantum head."""
        cls_out = self.classical(x)          # shape (batch, 2)
        q_out = self.quantum_head(cls_out)   # shape (batch, 1)
        return q_out.squeeze(-1)

# --------------------------------------------------------------------------- #
# Dataset utilities (borrowed from the quantum regression seed)
# --------------------------------------------------------------------------- #

def generate_superposition_data(num_wires: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic superposition states and labels."""
    omega_0 = np.zeros(2 ** num_wires, dtype=complex)
    omega_0[0] = 1.0
    omega_1 = np.zeros(2 ** num_wires, dtype=complex)
    omega_1[-1] = 1.0

    thetas = 2 * np.pi * np.random.rand(samples)
    phis = 2 * np.pi * np.random.rand(samples)
    states = np.zeros((samples, 2 ** num_wires), dtype=complex)
    for i in range(samples):
        states[i] = np.cos(thetas[i]) * omega_0 + np.exp(1j * phis[i]) * np.sin(thetas[i]) * omega_1
    labels = np.sin(2 * thetas) * np.cos(phis)
    return states, labels.astype(np.float32)

class HybridRegressionDataset(torch.utils.data.Dataset):
    """Dataset yielding superposition states and regression targets."""
    def __init__(self, samples: int, num_wires: int):
        self.states, self.labels = generate_superposition_data(num_wires, samples)

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.states)

    def __getitem__(self, index: int):  # type: ignore[override]
        return {
            "states": torch.tensor(self.states[index], dtype=torch.cfloat),
            "target": torch.tensor(self.labels[index], dtype=torch.float32),
        }

__all__ = [
    "FraudLayerParameters",
    "build_fraud_detection_program",
    "FraudDetectionRegressionHybrid",
    "HybridRegressionDataset",
    "generate_superposition_data",
]
