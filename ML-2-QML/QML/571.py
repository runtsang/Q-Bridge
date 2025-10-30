"""
Quantum implementation of the fraud‑detection model using PennyLane.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pennylane as qml
from pennylane import numpy as np


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
    """Clip a scalar to the interval [-bound, bound]."""
    return max(-bound, min(bound, value))


class FraudDetectionModel:
    """
    Variational photonic circuit for fraud detection.

    Parameters
    ----------
    layers : Iterable[FraudLayerParameters]
        Sequence of layer parameters. The first element is treated as the
        input layer, subsequent elements are hidden layers.
    device_name : str
        PennyLane device to use (e.g., 'default.qubit', 'lightning.qubit').
    wires : int
        Number of qubits (default 2).
    adaptive_measurement : bool
        Whether to use adaptive measurement of photon number after each layer.
    """

    def __init__(
        self,
        layers: Iterable[FraudLayerParameters],
        *,
        device_name: str = "default.qubit",
        wires: int = 2,
        adaptive_measurement: bool = False,
    ) -> None:
        self.layers = list(layers)
        if not self.layers:
            raise ValueError("At least one layer must be provided.")
        self.device = qml.device(device_name, wires=wires)
        self.adaptive_measurement = adaptive_measurement
        self.qnode = qml.QNode(self._circuit, self.device, interface="autograd")

    def _apply_layer(self, params: FraudLayerParameters, clip: bool = False) -> None:
        """Append gates corresponding to a single layer."""
        # Beam splitter
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        # Phase rotations
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        # Squeezing
        for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
            r_val = _clip(r, 5.0) if clip else r
            qml.Sgate(r_val, phi, wires=i)
        # Another beam splitter
        qml.BSgate(params.bs_theta, params.bs_phi, wires=[0, 1])
        # Phase rotations again
        for i, phase in enumerate(params.phases):
            qml.Rgate(phase, wires=i)
        # Displacement
        for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
            r_val = _clip(r, 5.0) if clip else r
            qml.Dgate(r_val, phi, wires=i)
        # Kerr
        for i, k in enumerate(params.kerr):
            k_val = _clip(k, 1.0) if clip else k
            qml.Kgate(k_val, wires=i)

    def _circuit(self, *params_flat: float) -> float:
        """
        Flattened parameters are reshaped into a list of FraudLayerParameters
        and used to construct the full circuit.
        """
        # Reconstruct parameters from flattened list
        # Each layer consumes 16 floats: bs_theta, bs_phi, 2 phases,
        # 2 squeeze_r, 2 squeeze_phi, 2 disp_r, 2 disp_phi, 2 kerr
        layer_params: list[FraudLayerParameters] = []
        idx = 0
        for _ in self.layers:
            bs_theta = params_flat[idx]
            bs_phi = params_flat[idx + 1]
            phases = (params_flat[idx + 2], params_flat[idx + 3])
            squeeze_r = (params_flat[idx + 4], params_flat[idx + 5])
            squeeze_phi = (params_flat[idx + 6], params_flat[idx + 7])
            disp_r = (params_flat[idx + 8], params_flat[idx + 9])
            disp_phi = (params_flat[idx + 10], params_flat[idx + 11])
            kerr = (params_flat[idx + 12], params_flat[idx + 13])
            layer_params.append(
                FraudLayerParameters(
                    bs_theta,
                    bs_phi,
                    phases,
                    squeeze_r,
                    squeeze_phi,
                    disp_r,
                    disp_phi,
                    kerr,
                )
            )
            idx += 14

        # Build the circuit
        for i, layer in enumerate(layer_params):
            self._apply_layer(layer, clip=(i > 0))
            if self.adaptive_measurement:
                # Example adaptive measurement: measure photon number and condition next gates
                # (here we simply ignore the result for brevity)
                qml.PauliZ(0, wires=0)

        # Output expectation of PauliZ on wire 0
        return qml.expval(qml.PauliZ(0))

    def predict(self, *param_values: float) -> np.ndarray:
        """
        Evaluate the circuit with given flattened parameter values.

        Returns
        -------
        expectation : float
            The expectation value of the measurement observable.
        """
        return self.qnode(*param_values)

    @classmethod
    def from_params(
        cls,
        params: Iterable[FraudLayerParameters],
        **kwargs,
    ) -> "FraudDetectionModel":
        """
        Convenience constructor mirroring the original build_fraud_detection_program.
        """
        return cls(params, **kwargs)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
