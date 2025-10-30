"""Quantum implementation of a fully‑connected layer via a variational circuit."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pennylane as qml

class FullyConnectedLayer:
    """
    A variational quantum circuit that maps a set of input features to a scalar
    expectation value.  The circuit consists of an input encoding with RX gates,
    followed by a stack of parameterised RY/RZ rotations and a ring of CNOTs.
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 1,
                 device: qml.Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device or qml.device("default.qubit", wires=n_qubits)
        self.params_shape = (n_layers, n_qubits, 2)  # (RY, RZ)
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: np.ndarray, params: np.ndarray) -> float:
            # Input encoding
            for i, x in enumerate(inputs):
                qml.RX(x, wires=i)
            # Parameterised layers
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.RY(params[layer, qubit, 0], wires=qubit)
                    qml.RZ(params[layer, qubit, 1], wires=qubit)
                # Ring entanglement
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Runs the circuit with the supplied input features.  The first ``n_qubits``
        elements of ``thetas`` are used as RX angles; missing values are padded
        with zeros.  Random circuit parameters are sampled each call.
        """
        inputs = np.array(thetas, dtype=np.float32)
        if len(inputs) < self.n_qubits:
            inputs = np.pad(inputs, (0, self.n_qubits - len(inputs)), mode="constant")
        else:
            inputs = inputs[: self.n_qubits]
        params = np.random.uniform(0, 2 * np.pi, self.params_shape)
        expectation = self.circuit(inputs, params)
        return np.array([expectation])

def FCL() -> FullyConnectedLayer:
    """
    Compatibility wrapper that returns a default instance of ``FullyConnectedLayer``.
    """
    return FullyConnectedLayer()

__all__ = ["FullyConnectedLayer", "FCL"]
