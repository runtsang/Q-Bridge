"""Quantum kernel that embeds data via a fraud‑detection inspired
pre‑encoding and a Quantum‑NAT style variational layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.functional import func_name_dict


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer used for a classical
    pre‑encoding of quantum data."""
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


class FraudLinear(tq.QuantumModule):
    """Linear transform applied classically before the quantum encoding."""
    def __init__(self, params: FraudLayerParameters, clip: bool = True) -> None:
        super().__init__()
        weight = torch.tensor(
            [[params.bs_theta, params.bs_phi],
             [params.squeeze_r[0], params.squeeze_r[1]]],
            dtype=torch.float32,
        )
        bias = torch.tensor(params.phases, dtype=torch.float32)
        if clip:
            weight = weight.clamp(-5.0, 5.0)
            bias = bias.clamp(-5.0, 5.0)
        self.linear = nn.Linear(2, 2)
        with torch.no_grad():
            self.linear.weight.copy_(weight)
            self.linear.bias.copy_(bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.linear(inputs)


class QLayer(tq.QuantumModule):
    """Variational sub‑module inspired by the Quantum‑NAT architecture."""
    def __init__(self, n_wires: int) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.random_layer = tq.RandomLayer(n_ops=50, wires=list(range(n_wires)))
        self.rx0 = tq.RX(has_params=True, trainable=True)
        self.ry0 = tq.RY(has_params=True, trainable=True)
        self.rz0 = tq.RZ(has_params=True, trainable=True)
        self.crx0 = tq.CRX(has_params=True, trainable=True)

    @tq.static_support
    def forward(self, qdev: tq.QuantumDevice) -> None:
        self.random_layer(qdev)
        self.rx0(qdev, wires=0)
        self.ry0(qdev, wires=1)
        self.rz0(qdev, wires=3)
        self.crx0(qdev, wires=[0, 2])
        tqf.hadamard(qdev, wires=3, static=self.static_mode, parent_graph=self.graph)
        tqf.sx(qdev, wires=2, static=self.static_mode, parent_graph=self.graph)
        tqf.cnot(qdev, wires=[3, 0], static=self.static_mode, parent_graph=self.graph)


class HybridQuantumKernel(tq.QuantumModule):
    """Quantum kernel that encodes classical data, applies a variational
    quantum layer, and returns the overlap amplitude."""
    def __init__(self,
                 n_wires: int = 4,
                 preproc_params: FraudLayerParameters | None = None) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=n_wires)
        self.preproc = FraudLinear(preproc_params, clip=False) if preproc_params else nn.Identity()
        self.encoder = tq.GeneralEncoder(tq.encoder_op_list_name_dict["4x4_ryzxy"])
        self.q_layer = QLayer(n_wires)
        self.measure = tq.MeasureAll(tq.PauliZ)
        self.norm = nn.BatchNorm1d(n_wires)

    @tq.static_support
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Pre‑encode classically
        x = self.preproc(x).view(-1, 2)
        y = self.preproc(y).view(-1, 2)

        # Encode x
        self.q_device.reset_states(x.shape[0])
        self.encoder(self.q_device, x)

        # Variational layer
        self.q_layer(self.q_device)

        # Encode -y (inverse) to probe similarity
        self.encoder(self.q_device, -y)

        out = self.measure(self.q_device)
        return torch.abs(out[:, 0])

def kernel_matrix(a: Sequence[torch.Tensor],
                  b: Sequence[torch.Tensor],
                  preproc_params: FraudLayerParameters | None = None) -> np.ndarray:
    kernel = HybridQuantumKernel(preproc_params=preproc_params)
    return np.array([[kernel(x, y).item() for y in b] for x in a])


__all__ = ["FraudLayerParameters", "HybridQuantumKernel", "kernel_matrix"]
