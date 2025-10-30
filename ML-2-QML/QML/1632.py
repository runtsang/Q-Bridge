"""Quantum fraud‑detection circuit using Pennylane."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import pennylane as qml
import torch
from pennylane import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


class FraudDetectionModel:
    """Quantum fraud‑detection pipeline implemented as a Pennylane QNode."""

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        device_name: str = "default.qubit",
        wires: int = 2,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.device = qml.device(device_name, wires=wires)
        self.qnode = qml.QNode(self._circuit, self.device, interface="torch")

    def _apply_layer(
        self,
        params: FraudLayerParameters,
        clip: bool = True,
    ) -> None:
        """Append a photonic layer of gates to the circuit."""
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_clipped = r if not clip else _clip(r, 5.0)
            qml.Sgate(r_clipped, phi, wires=i)
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_clipped = r if not clip else _clip(r, 5.0)
            qml.Dgate(r_clipped, phi, wires=i)
        for i, k in enumerate(params.kerr):
            k_clipped = k if not clip else _clip(k, 1.0)
            qml.Kgate(k_clipped, wires=i)

    def _circuit(self, inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Variational circuit that mirrors the classical layers."""
        # Encode classical inputs into displacement gates
        qml.Dgate(inputs[0], 0.0, wires=0)
        qml.Dgate(inputs[1], 0.0, wires=1)

        # Apply input photonic layer
        self._apply_layer(self.input_params, clip=False)

        # Apply subsequent layers
        for layer in self.layers:
            self._apply_layer(layer, clip=True)

        # Final measurement: expectation of PauliZ on first qubit
        return qml.expval(qml.PauliZ(0))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning a fraud probability."""
        # Expectation values are in [-1, 1]; map to [0, 1]
        raw = self.qnode(x.numpy(), torch.tensor([]))
        return (raw + 1.0) / 2.0

    def evaluate(self, dataloader) -> Tuple[float, float]:
        """Compute mean squared error and accuracy on a dataset."""
        total_loss = 0.0
        correct, total = 0, 0
        for xb, yb in dataloader:
            preds = self.predict(xb).squeeze()
            loss = torch.mean((preds - yb) ** 2).item()
            total_loss += loss * xb.size(0)
            preds_bin = (preds > 0.5).float()
            correct += (preds_bin == yb).sum().item()
            total += xb.size(0)
        return total_loss / total, correct / total

    def __repr__(self) -> str:
        return f"<FraudDetectionModel quantum layers={len(self.layers)}>"
__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
