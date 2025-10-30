"""FraudDetectionModel: Quantum feature extractor based on a PennyLane variational circuit.

This module extends the original photonic circuit by converting it into a variational ansatz
that can be trained as a quantum kernel or feature map. The circuit operates on two qubits,
uses parameterized rotations and entangling gates, and outputs expectation values of Pauli‑Z
operators as a feature vector. The parameters are stored in a FraudLayerParameters dataclass
that mirrors the classical counterpart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import pennylane as qml
import numpy as np


@dataclass
class FraudLayerParameters:
    """Parameters describing a single photonic layer, mirroring the classical schema."""
    bs_theta: float
    bs_phi: float
    phases: tuple[float, float]
    squeeze_r: tuple[float, float]
    squeeze_phi: tuple[float, float]
    displacement_r: tuple[float, float]
    displacement_phi: tuple[float, float]
    kerr: tuple[float, float]
    activation: str = "tanh"  # retained for compatibility; not used in the quantum circuit
    dropout_rate: float = 0.0  # retained for compatibility; not used in the quantum circuit


class FraudDetectionModel:
    """Quantum feature extractor based on a variational circuit."""

    def __init__(self, dev: qml.Device | None = None, shots: int = 1024):
        self.dev = dev or qml.device("default.qubit", wires=2, shots=shots)

    def _build_ansatz(self, params: FraudLayerParameters) -> qml.QNode:
        @qml.qnode(self.dev)
        def circuit(inputs: np.ndarray) -> np.ndarray:
            # Encode the 2‑dimensional classical input as rotations on each qubit
            qml.RX(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)

            # Layer 1: parameterized rotations and entangling gates
            qml.RZ(params.bs_theta, wires=0)
            qml.RZ(params.bs_phi, wires=1)
            qml.CZ(wires=[0, 1])

            # Phase and squeezing (approximated by small rotations)
            for i, phase in enumerate(params.phases):
                qml.RZ(phase, wires=i)
            for i, (r, phi) in enumerate(zip(params.squeeze_r, params.squeeze_phi)):
                qml.RX(r * np.cos(phi), wires=i)
                qml.RY(r * np.sin(phi), wires=i)

            # Displacement (simulated by additional rotations)
            for i, (r, phi) in enumerate(zip(params.displacement_r, params.displacement_phi)):
                qml.RX(r * np.cos(phi), wires=i)
                qml.RY(r * np.sin(phi), wires=i)

            # Kerr (approximated by a cubic phase)
            for i, k in enumerate(params.kerr):
                qml.RZ(k * np.abs(inputs[i]) ** 3, wires=i)

            # Measurement: expectation values of Pauli‑Z on each qubit
            return [qml.expval(qml.PauliZ(i)) for i in range(2)]

        return circuit

    def sample_features(self, inputs: np.ndarray, params: FraudLayerParameters) -> np.ndarray:
        """Return quantum‑generated features for a batch of classical inputs."""
        circuit = self._build_ansatz(params)
        return np.array([circuit(inp) for inp in inputs])

    def train_ansatz(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        params: FraudLayerParameters,
        lr: float = 0.01,
        epochs: int = 100,
    ):
        """Simple training loop to adjust the ansatz parameters to fit the labels."""
        circuit = self._build_ansatz(params)
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            loss = 0.0
            for inp, lbl in zip(data, labels):
                pred = circuit(inp)[0]  # use first qubit expectation as output
                loss += (pred - lbl) ** 2
            loss /= len(data)
            opt.step(circuit, grad_fn=circuit.grad)
        return circuit

    def __call__(self, inputs: np.ndarray, params: FraudLayerParameters) -> np.ndarray:
        """Convenience wrapper to get features."""
        return self.sample_features(inputs, params)


__all__ = ["FraudLayerParameters", "FraudDetectionModel"]
