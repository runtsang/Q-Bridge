"""Quantum photonic fraud detection model using PennyLane's Gaussian device."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pennylane as qml

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


def _apply_layer(wires: Sequence[int], params: FraudLayerParameters, *, clip: bool) -> None:
    """Apply a photonic layer to the given wires."""
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
        r_val = _clip(r, 5) if clip else r
        qml.Sgate(r_val, phi, wires=wires[i])
    qml.BSgate(params.bs_theta, params.bs_phi, wires=wires)
    for i, phase in enumerate(params.phases):
        qml.Rgate(phase, wires=wires[i])
    for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
        r_val = _clip(r, 5) if clip else r
        qml.Dgate(r_val, phi, wires=wires[i])
    for i, k in enumerate(params.kerr):
        k_val = _clip(k, 1) if clip else k
        qml.Kgate(k_val, wires=wires[i])


class FraudDetectionHybrid:
    """Hybrid quantum photonic model that embeds classical pre‑processing."""

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.dev = qml.device("default.gaussian", wires=2)
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev)
        def circuit(inputs: np.ndarray) -> np.ndarray:
            # Classical linear pre‑processing
            weight = np.array(
                [
                    [self.input_params.bs_theta, self.input_params.bs_phi],
                    [self.input_params.squeeze_r[0], self.input_params.squeeze_r[1]],
                ],
                dtype=float,
            )
            bias = np.array(self.input_params.phases, dtype=float)
            x = np.tanh(weight @ inputs + bias)
            x = x * np.array(self.input_params.displacement_r) + np.array(self.input_params.displacement_phi)

            # Encode the pre‑processed vector as displacements
            for i in range(2):
                qml.Dgate(x[i], 0.0, wires=i)

            # Photonic layers
            _apply_layer([0, 1], self.input_params, clip=False)
            for layer in self.layers:
                _apply_layer([0, 1], layer, clip=True)

            # Measurement
            return qml.expval(qml.PauliZ(0))
        return circuit

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self._circuit(inputs)

    def train_step(self, optimizer, loss_fn, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Perform one optimisation step using PennyLane's autograd."""
        preds = self._circuit(inputs)
        loss = loss_fn(preds, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
