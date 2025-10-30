"""Quantum fraud detection model built with PennyLane."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pennylane as qml
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer in the quantum model."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]
    trainable: bool = False


class FraudDetectionModel:
    """Variational quantum circuit that emulates the photonic fraud‑detection architecture.

    Parameters
    ----------
    layer_params
        Iterable of :class:`FraudLayerParameters` describing each layer.
    dev
        PennyLane device name (e.g. ``'default.qubit'`` or a Braket backend).
    wires
        Number of photonic modes; default is 2.
    """
    def __init__(
        self,
        layer_params: Iterable[FraudLayerParameters],
        dev: str = "default.qubit",
        wires: int = 2,
    ) -> None:
        self.dev = qml.device(dev, wires=wires)
        self.layer_params: List[FraudLayerParameters] = list(layer_params)
        self._qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, inputs: np.ndarray) -> np.ndarray:
        """Encode the 2‑dimensional classical input and apply all photonic layers."""
        qml.QubitStateVector(inputs, wires=range(2))
        for params in self.layer_params:
            self._apply_layer(params)
        return qml.expval(qml.PauliZ(0))

    def _apply_layer(self, params: FraudLayerParameters) -> None:
        """Apply a single photonic layer to the circuit."""
        # Beam‑splitter
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])

        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)

        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            qml.Sgate(r if not params.trainable else self._clip(r, 5), phi, wires=i)

        # Second beam‑splitter
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])

        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)

        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            qml.Dgate(r if not params.trainable else self._clip(r, 5), phi, wires=i)

        for i, k in enumerate(params.kerr):
            qml.Kgate(k if not params.trainable else self._clip(k, 1), wires=i)

    @staticmethod
    def _clip(value: float, bound: float) -> float:
        return max(-bound, min(bound, value))

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        """Evaluate the quantum circuit on a batch of 2‑dimensional data."""
        return self._qnode(inputs)

    def trainable_parameters(self) -> List[float]:
        """Return a list of parameters that would be optimised if the circuit were made trainable."""
        params: List[float] = []
        for layer in self.layer_params:
            if layer.trainable:
                params.extend(
                    [
                        layer.bs_theta,
                        layer.bs_phi,
                        *layer.phases,
                        *layer.squeeze_r,
                        *layer.squeeze_phi,
                        *layer.displacement_r,
                        *layer.displacement_phi,
                        *layer.kerr,
                    ]
                )
        return params


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
