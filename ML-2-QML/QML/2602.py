"""Hybrid fraud detection and kernel module (quantum implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

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


def _layer_ansatz(params: FraudLayerParameters) -> list[dict]:
    """
    Translate a photonic layer into a list of quantum gate descriptors.
    Each descriptor contains the gate name, target wires, and a list of
    input indices that will be used as rotation angles.
    """
    ansatz = []

    # Encode beam‑splitter angles as Ry rotations
    ansatz.append({"input_idx": [0], "func": "ry", "wires": [0]})
    ansatz.append({"input_idx": [1], "func": "ry", "wires": [1]})

    # Encode phases as Rz rotations
    for i, _ in enumerate(params.phases):
        ansatz.append({"input_idx": [i], "func": "rz", "wires": [i]})

    # Encode squeezing and displacement via additional Ry rotations
    for i, _ in enumerate(params.squeeze_r):
        ansatz.append({"input_idx": [i], "func": "ry", "wires": [i]})
    for i, _ in enumerate(params.displacement_r):
        ansatz.append({"input_idx": [i], "func": "ry", "wires": [i]})

    # Encode Kerr non‑linearity as Rz
    for i, _ in enumerate(params.kerr):
        ansatz.append({"input_idx": [i], "func": "rz", "wires": [i]})

    # Add a simple entangling gate
    ansatz.append({"input_idx": [], "func": "cx", "wires": [0, 1]})

    return ansatz


def _combine_ansatz(layers: Iterable[FraudLayerParameters]) -> list[dict]:
    """Concatenate ansatzes from all layers."""
    ansatz: list[dict] = []
    for layer in layers:
        ansatz.extend(_layer_ansatz(layer))
    return ansatz


class HybridFraudKernel(tq.QuantumModule):
    """
    Quantum kernel that encodes fraud‑detection parameters into a variational
    circuit and evaluates the overlap between two input states.
    """

    def __init__(
        self,
        fraud_input_params: FraudLayerParameters,
        fraud_layers: Iterable[FraudLayerParameters],
        n_wires: int = 4,
    ) -> None:
        super().__init__()
        self.n_wires = n_wires
        self.q_device = tq.QuantumDevice(n_wires=self.n_wires)
        self.ansatz = _combine_ansatz([fraud_input_params] + list(fraud_layers))

    @tq.static_support
    def forward(self, q_device: tq.QuantumDevice, x: torch.Tensor, y: torch.Tensor) -> None:
        """Encode x and y and compute the overlap."""
        q_device.reset_states(x.shape[0])

        # Encode x
        for info in self.ansatz:
            params = (
                x[:, info["input_idx"]]
                if info["input_idx"]
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

        # Encode y with negative parameters
        for info in reversed(self.ansatz):
            params = (
                -y[:, info["input_idx"]]
                if info["input_idx"]
                else None
            )
            func_name_dict[info["func"]](q_device, wires=info["wires"], params=params)

    def kernel_matrix(
        self, a: Sequence[torch.Tensor], b: Sequence[torch.Tensor]
    ) -> np.ndarray:
        """Return the Gram matrix between two datasets."""
        kernel = np.empty((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                self.forward(self.q_device, x, y)
                kernel[i, j] = torch.abs(self.q_device.states.view(-1)[0]).item()
        return kernel


__all__ = ["FraudLayerParameters", "HybridFraudKernel"]
