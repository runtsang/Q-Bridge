"""Core circuit factory for the incremental data-uploading classifier with a hybrid quantum‑classical interface."""

from __future__ import annotations

import pennylane as qml
from pennylane import numpy as np
from typing import Iterable, List


class QuantumClassifierModel:
    """
    Variational classifier implemented with Pennylane.
    Mirrors the classical factory while providing quantum‑centric features:
    - Feature‑map encoding with RX rotations
    - Parameter‑shiftable variational layers
    - Automatic gradient computation via autograd
    """

    def __init__(self, num_qubits: int, depth: int = 2, device_name: str = "default.qubit"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.device = qml.device(device_name, wires=num_qubits)

        # Parameter vectors
        self.enc_params = np.zeros(num_qubits, requires_grad=False)
        self.var_params = np.zeros(num_qubits * depth, requires_grad=True)

        # Observables (single‑qubit Z on each wire)
        self._observables = [qml.PauliZ(i) for i in range(num_qubits)]

        # Build QNode
        self._qnode = qml.QNode(self._circuit, self.device, interface="autograd")

    def _circuit(self, *params):
        # Unpack parameters
        enc = params[:self.num_qubits]
        var = params[self.num_qubits:]

        # Feature encoding
        for i in range(self.num_qubits):
            qml.RX(enc[i], wires=i)

        # Variational layers
        idx = 0
        for _ in range(self.depth):
            for i in range(self.num_qubits):
                qml.RY(var[idx], wires=i)
                idx += 1
            for i in range(self.num_qubits - 1):
                qml.CZ(i, i + 1)

        # Measurement
        return [qml.expval(obs) for obs in self._observables]

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Compute expectation values for a batch of inputs.
        x is expected to be a 2‑D array of shape (batch, num_qubits).
        """
        preds = []
        for sample in x:
            params = np.concatenate([sample, self.var_params])
            preds.append(self._qnode(*params))
        return np.array(preds)

    @property
    def encoding(self) -> Iterable[int]:
        """List of wire indices used for encoding."""
        return list(range(self.num_qubits))

    @property
    def weight_sizes(self) -> Iterable[int]:
        """Number of trainable parameters in the variational part."""
        return [self.var_params.size]

    @property
    def observables(self) -> Iterable[qml.operation.Operator]:
        """PauliZ observables on each qubit."""
        return self._observables

    def get_params(self) -> np.ndarray:
        """Return current variational parameters."""
        return self.var_params

    def set_params(self, new_params: np.ndarray):
        """Set variational parameters (useful for external optimizers)."""
        self.var_params = new_params


def build_classifier_circuit(num_qubits: int, depth: int = 2, device_name: str = "default.qubit") -> QuantumClassifierModel:
    return QuantumClassifierModel(num_qubits, depth, device_name)


__all__ = ["QuantumClassifierModel", "build_classifier_circuit"]
