"""Quantum classifier using Pennylane with amplitude encoding and variational layers."""
from __future__ import annotations

from typing import Iterable, List, Tuple
import pennylane as qml
import pennylane.numpy as np
import torch


class QuantumClassifier:
    """
    Variational quantum classifier that mirrors the classical helper interface.
    Parameters
    ----------
    num_qubits: int
        Number of qubits / input features.
    depth: int
        Number of variational layers.
    feature_map: str, optional
        Type of feature map ('ry', 'rycz', 'amplitude').
    entanglement: str, optional
        Entanglement pattern for the variational layers ('full', 'circular').
    """

    def __init__(self, num_qubits: int, depth: int,
                 feature_map: str = "ry", entanglement: str = "circular"):
        self.num_qubits = num_qubits
        self.depth = depth
        self.feature_map = feature_map
        self.entanglement = entanglement
        self._device = qml.device("default.qubit", wires=self.num_qubits)
        self.circuit, self.encoding, self.weight_sizes, self.observables = self._build_circuit()

    def _build_circuit(self) -> Tuple[qml.QNode, List[int], List[int], List[qml.operation.Operator]]:
        def feature_map_fn(x):
            if self.feature_map == "ry":
                for i, xi in enumerate(x):
                    qml.RY(xi, wires=i)
            elif self.feature_map == "rycz":
                for i, xi in enumerate(x):
                    qml.RY(xi, wires=i)
                for i in range(self.num_qubits - 1):
                    qml.CZ(wires=[i, i + 1])
            elif self.feature_map == "amplitude":
                qml.AmplitudeEmbedding(features=x, wires=range(self.num_qubits), normalize=True)
            else:
                raise ValueError(f"Unsupported feature map {self.feature_map}")

        def variational_layer(params):
            for i, p in enumerate(params):
                qml.RY(p, wires=i)
            if self.entanglement == "full":
                for i in range(self.num_qubits):
                    for j in range(i + 1, self.num_qubits):
                        qml.CZ(wires=[i, j])
            elif self.entanglement == "circular":
                for i in range(self.num_qubits):
                    qml.CZ(wires=[i, (i + 1) % self.num_qubits])

        @qml.qnode(self._device, interface="torch")
        def circuit(x, params):
            feature_map_fn(x)
            for layer in range(self.depth):
                variational_layer(params[layer])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        weight_sizes = [self.num_qubits] * self.depth
        encoding = list(range(self.num_qubits))
        observables = [qml.PauliZ(i) for i in range(self.num_qubits)]

        return circuit, encoding, weight_sizes, observables

    def get_circuit(self):
        """Return the Pennylane QNode."""
        return self.circuit

    def get_encoding(self):
        """Indices of the input features used in the circuit."""
        return self.encoding

    def get_weight_sizes(self):
        """Number of parameters per variational layer."""
        return self.weight_sizes

    def get_observables(self):
        """PauliZ observables measured on each qubit."""
        return self.observables

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(num_qubits={self.num_qubits}, depth={self.depth}, "
                f"feature_map={self.feature_map}, entanglement={self.entanglement})")


__all__ = ["QuantumClassifier"]
