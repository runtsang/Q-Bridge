from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import torch
import torchquantum as tq
from torchquantum.functional import func_name_dict


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
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


class KernalAnsatz(tq.QuantumModule):
    """Quantum ansatz that encodes fraud‑detection parameters and data."""
    def __init__(self, params: List[FraudLayerParameters]) -> None:
        super().__init__()
        self.params = params

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        # Encode forward pass with data
        for layer in self.params:
            for wire in range(2):
                # Data‑dependent rotations
                theta = layer.bs_theta if wire == 0 else layer.bs_phi
                phi = layer.phases[wire]
                # Apply Ry and Rz gates with data‑scaled parameters
                func_name_dict["ry"](q_device, wires=[wire], params=x[:, wire] * theta)
                func_name_dict["rz"](q_device, wires=[wire], params=x[:, wire] * phi)
            # Entangling gate
            func_name_dict["cx"](q_device, wires=[0, 1])
        # Reverse pass with negative data
        for layer in reversed(self.params):
            for wire in range(2):
                theta = layer.bs_theta if wire == 0 else layer.bs_phi
                phi = layer.phases[wire]
                func_name_dict["ry"](q_device, wires=[wire], params=-y[:, wire] * theta)
                func_name_dict["rz"](q_device, wires=[wire], params=-y[:, wire] * phi)
            func_name_dict["cx"](q_device, wires=[0, 1])


class HybridFraudKernel(tq.QuantumModule):
    """Quantum kernel that mirrors the fraud‑detection architecture."""
    def __init__(self, params: List[FraudLayerParameters]) -> None:
        super().__init__()
        self.n_wires = 2
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = KernalAnsatz(params)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        self.ansatz(self.q_device, x, y)
        # Return absolute value of first element of the state vector
        return torch.abs(self.q_device.states.view(-1)[0])

    def kernel_matrix(self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]) -> np.ndarray:
        """Compute Gram matrix between two sequences of tensors."""
        return np.array([[self.forward(x, y).item() for y in b] for x in a])


__all__ = ["FraudLayerParameters", "HybridFraudKernel"]
