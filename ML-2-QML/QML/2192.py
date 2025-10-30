"""
Hybrid quantum‑classical fraud‑detection model using PennyLane.
The quantum part is a variational circuit producing two expectation values that
are fed into a simple classical readout.
"""

from __future__ import annotations

import pennylane as qml
import numpy as np
from dataclasses import dataclass
from typing import Iterable, Tuple


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic layer, retained for consistency."""
    bs_theta: float
    bs_phi: float
    phases: Tuple[float, float]
    squeeze_r: Tuple[float, float]
    squeeze_phi: Tuple[float, float]
    displacement_r: Tuple[float, float]
    displacement_phi: Tuple[float, float]
    kerr: Tuple[float, float]


class FraudDetectionHybrid:
    """
    Hybrid quantum‑classical fraud‑detection model.
    The quantum circuit is a simple variational network that outputs two
    expectation values; a lightweight classical readout maps them to a logit.

    Parameters
    ----------
    input_params : FraudLayerParameters
        Parameters for the first layer.
    layers : Iterable[FraudLayerParameters]
        Additional layers.
    n_shots : int, optional
        Number of shots for the default.qubit simulator. Defaults to 1000.
    """

    def __init__(
        self,
        input_params: FraudLayerParameters,
        layers: Iterable[FraudLayerParameters],
        n_shots: int = 1000,
    ) -> None:
        self.device = qml.device("default.qubit", wires=2, shots=n_shots)
        self.params = [input_params] + list(layers)
        self.q_params = self._flatten_params()
        self.quantum_circuit = qml.QNode(self._quantum_circuit, self.device, interface="autograd")

    def _flatten_params(self) -> np.ndarray:
        """Flatten the dataclass parameters into a 1‑D array for the variational circuit."""
        flat = []
        for p in self.params:
            flat.extend(
                [p.bs_theta, p.bs_phi]
                + list(p.phases)
                + list(p.squeeze_r)
                + list(p.squeeze_phi)
                + list(p.displacement_r)
                + list(p.displacement_phi)
                + list(p.kerr)
            )
        return np.array(flat, dtype=np.float64)

    def _quantum_circuit(self, *qml_params):
        """Variational circuit that mimics the layered photonic structure."""
        idx = 0
        for _ in self.params:
            theta = qml_params[idx]
            phi = qml_params[idx + 1]
            idx += 2

            # Beam‑splitter analogue: entangle the two wires
            qml.CNOT(wires=[0, 1])

            # Local phases
            for i in range(2):
                qml.RZ(qml_params[idx + i], wires=i)
            idx += 2

            # Squeezing analogue: simple rotation
            for i in range(2):
                qml.RX(qml_params[idx + i], wires=i)
            idx += 2

            # Displacement and Kerr terms as phase shifts
            for i in range(2):
                qml.PhaseShift(qml_params[idx + i], wires=i)
            idx += 2

            for i in range(2):
                qml.PhaseShift(qml_params[idx + i], wires=i)
            idx += 2

        # Return two expectation values
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    def _classical_readout(self, z0: float, z1: float) -> float:
        """Linear readout mapping quantum expectations to a logit."""
        return 0.5 * z0 - 0.5 * z1 + 0.1

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: ignore classical input `x` (kept for API compatibility)
        and return the predicted fraud probability.
        """
        z0, z1 = self.quantum_circuit(*self.q_params)
        logit = self._classical_readout(z0, z1)
        prob = 1 / (1 + np.exp(-logit))
        return prob

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
