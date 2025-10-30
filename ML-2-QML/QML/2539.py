"""Quantum hybrid fraud detection model combining a quantum convolutional filter
and photonic‑inspired variational circuit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pennylane as qml


@dataclass
class FraudLayerParameters:
    """Parameters describing a photonic‑inspired variational layer."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]


def _clip(value: float, bound: float) -> float:
    """Clip a scalar to a symmetric bound."""
    return max(-bound, min(bound, value))


class FraudDetectionHybrid:
    """
    Quantum hybrid fraud detection model.

    1. Quantum convolutional circuit (4 qubits) processes 2x2 patches.
    2. Classical linear reduction maps the 4*14*14 feature vector to 2 values.
    3. Two‑qubit photonic variational circuit outputs a scalar.
    """

    def __init__(self, input_params: FraudLayerParameters, layers: Iterable[FraudLayerParameters]) -> None:
        self.input_params = input_params
        self.layers = list(layers)

        # Devices
        self.conv_dev = qml.device("default.qubit", wires=4)
        self.photon_dev = qml.device("default.qubit", wires=2)

        # QNodes
        self.conv_qnode = qml.QNode(self._conv_circuit, self.conv_dev)
        self.photon_qnode = qml.QNode(self._photon_circuit, self.photon_dev)

        # Classical reduction matrix (random for illustration)
        self.reduce_matrix = np.random.randn(2, 4 * 14 * 14)

    def _conv_circuit(self, image: np.ndarray) -> np.ndarray:
        """
        Quantum convolutional circuit that processes 2x2 patches.
        Returns a 196*4 dimensional feature vector.
        """
        features = []
        for r in range(0, 28, 2):
            for c in range(0, 28, 2):
                patch = image[0, 0, r:r + 2, c:c + 2]  # shape (2, 2)
                # Encode patch into 4 qubits
                qml.RY(patch[0, 0], wires=0)
                qml.RY(patch[0, 1], wires=1)
                qml.RY(patch[1, 0], wires=2)
                qml.RY(patch[1, 1], wires=3)
                # Random two‑qubit layer
                qml.CNOT(0, 1)
                qml.CNOT(1, 2)
                qml.CNOT(2, 3)
                qml.CNOT(3, 0)
                qml.CNOT(0, 2)
                qml.CNOT(1, 3)
                qml.CNOT(0, 3)
                qml.CNOT(1, 2)
                # Measure each qubit
                for i in range(4):
                    features.append(qml.expval(qml.PauliZ(i)))
        return np.array(features)

    @staticmethod
    def _photonic_layer(params: FraudLayerParameters, wires: Sequence[int]) -> None:
        """Apply a photonic‑inspired variational layer on two qubits."""
        # Beam‑splitter (approximated by CNOT)
        qml.CNOT(wires[0], wires[1])
        # Phases
        qml.RZ(params.bs_theta, wires=wires[0])
        qml.RZ(params.bs_phi, wires=wires[1])
        qml.RZ(params.phases[0], wires=wires[0])
        qml.RZ(params.phases[1], wires=wires[1])
        # Squeezing
        qml.RZ(params.squeeze_r[0], wires=wires[0])
        qml.RZ(params.squeeze_r[1], wires=wires[1])
        # Displacement
        qml.RZ(params.displacement_r[0], wires=wires[0])
        qml.RZ(params.displacement_r[1], wires=wires[1])
        # Kerr non‑linearity (approximated by CZ)
        qml.CZ(wires[0], wires[1])

    def _photon_circuit(self, reduced: np.ndarray) -> float:
        """
        Two‑qubit photonic circuit that consumes the reduced classical features.
        Returns a scalar expectation value.
        """
        # Encode reduced features as rotations
        qml.RY(reduced[0], wires=0)
        qml.RY(reduced[1], wires=1)
        # Apply each photonic layer
        for params in self.layers:
            self._photonic_layer(params, wires=[0, 1])
        # Output observable
        return qml.expval(qml.PauliZ(0))

    def __call__(self, image: np.ndarray) -> float:
        """
        Run the full hybrid model on a single 28×28 image.

        Parameters
        ----------
        image : np.ndarray
            Array of shape (1, 1, 28, 28) with pixel intensities in [0, 1].

        Returns
        -------
        float
            The model's scalar output.
        """
        # Quantum convolution
        conv_features = self.conv_qnode(image)  # shape (196*4,)
        # Classical reduction
        reduced = self.reduce_matrix @ conv_features  # shape (2,)
        # Photonic circuit
        return float(self.photon_qnode(reduced))


__all__ = ["FraudLayerParameters", "FraudDetectionHybrid"]
