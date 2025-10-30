"""Quantum implementation of fraud detection using PennyLane.

The `FraudModel` class builds a variational circuit that mirrors the
photonic layer structure.  It uses parameterised beamsplitters,
squeezers, displacements and Kerr gates.  The class is API compatible
with the classical counterpart and can be trained with gradient-based
optimisation.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np


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


class FraudModel:
    """A PennyLane circuit that reproduces the photonic fraud detection
    model.  It accepts the same `FraudLayerParameters` as the
    classical version.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first (input) layer.
    layers : Iterable[FraudLayerParameters]
        Parameters for all subsequent layers.
    clip : bool, default=True
        Whether to clip the parameters to a bounded range.
    dev : pennylane.Device, optional
        Quantum device to use.  Defaults to ``default.qubit`` with
        two wires.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        clip: bool = True,
        dev: qml.Device | None = None,
    ) -> None:
        self.input_params = input_params
        self.layers = list(layers)
        self.clip = clip
        self.dev = dev or qml.device("default.qubit", wires=2)

    def _apply_layer(self, params: FraudLayerParameters, clip: bool) -> None:
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_val = r if not clip else _clip(r, 5)
            qml.Sgate(r_val, phi, wires=i)
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_val = r if not clip else _clip(r, 5)
            qml.Dgate(r_val, phi, wires=i)
        for i, k in enumerate(params.kerr):
            k_val = k if not clip else _clip(k, 1)
            qml.Kgate(k_val, wires=i)

    def evaluate(self) -> np.ndarray:
        """Run the circuit and return Pauli‑Z expectation values on both wires.

        Returns
        -------
        np.ndarray
            Array of shape (2,) containing the expectation values.
        """

        @qml.qnode(self.dev)
        def circuit():
            self._apply_layer(self.input_params, clip=False)
            for layer in self.layers:
                self._apply_layer(layer, clip=self.clip)
            return [
                qml.expval(qml.PauliZ(0)),
                qml.expval(qml.PauliZ(1)),
            ]

        return circuit()


__all__ = ["FraudLayerParameters", "FraudModel"]
