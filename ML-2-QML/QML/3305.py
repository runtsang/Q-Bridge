"""Quantum kernel mirroring the fraud‑detection inspired ansatz
using TorchQuantum and data re‑uploading."""

from __future__ import annotations

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import ry, rz, cx
from dataclasses import dataclass
from typing import Sequence, Iterable, List

@dataclass
class FraudLayerParameters:
    """Parameters describing a quantum fraud‑detection layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]

class FraudQuantumLayer(tq.QuantumModule):
    """Single layer that encodes data via rotations, entanglement,
    and re‑uploading, parameterised by FraudLayerParameters."""
    def __init__(self, params: FraudLayerParameters, n_wires: int = 2) -> None:
        super().__init__()
        self.params = params
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        q_device.reset_states(x.shape[0])

        # Encode x with rotations and phase gates
        for i in range(self.n_wires):
            angle = self.params.bs_theta * x[:, i] + self.params.squeeze_r[i]
            ry(q_device, wires=i, params=angle)
            rz(q_device, wires=i, params=self.params.phases[i])

        # Entanglement
        cx(q_device, wires=[0, 1])

        # Re‑upload negative y
        for i in range(self.n_wires):
            angle = -self.params.bs_phi * y[:, i] + self.params.displacement_r[i]
            ry(q_device, wires=i, params=angle)
            rz(q_device, wires=i, params=self.params.displacement_phi[i])

class HybridQuantumKernel(tq.QuantumModule):
    """Quantum kernel that chains multiple fraud‑detection inspired layers
    and returns the absolute overlap of the first basis state."""
    def __init__(
        self,
        fraud_params: List[FraudLayerParameters] | None = None,
        n_wires: int = 2,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.layers = tq.nn.ModuleList()
        if fraud_params:
            for p in fraud_params:
                self.layers.append(FraudQuantumLayer(p, n_wires))
        else:
            # Default layer if none supplied
            default = FraudLayerParameters(
                bs_theta=0.5, bs_phi=0.5,
                phases=(0.0, 0.0),
                squeeze_r=(0.0, 0.0),
                squeeze_phi=(0.0, 0.0),
                displacement_r=(0.0, 0.0),
                displacement_phi=(0.0, 0.0),
                kerr=(0.0, 0.0),
            )
            self.layers.append(FraudQuantumLayer(default, n_wires))

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            layer(self.q_device, x, y)
        # Return absolute overlap of |00> with the final state
        return torch.abs(self.q_device.states.view(-1)[0])

def kernel_matrix(
    a: Sequence[torch.Tensor],
    b: Sequence[torch.Tensor],
    fraud_params: List[FraudLayerParameters] | None = None,
) -> np.ndarray:
    model = HybridQuantumKernel(fraud_params)
    return np.array([[model(x, y).item() for y in b] for x in a])

__all__ = ["FraudLayerParameters", "HybridQuantumKernel", "kernel_matrix"]
