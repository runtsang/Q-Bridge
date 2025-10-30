"""Quantum fully connected layer using Pennylane variational circuit.

This class extends the original seed by providing a multi-qubit variational
circuit with configurable entanglement and layer depth. It exposes a ``run``
method that accepts a batch of parameter sets and returns expectation values,
mirroring the API of the original seed.
"""

import pennylane as qml
import numpy as np
from typing import Iterable, List, Tuple, Union

class FullyConnectedLayer:
    """
    Parameterized quantum circuit that emulates a fully connected layer.
    Supports multiple qubits and a customizable entanglement pattern.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        n_layers: int = 1,
        entanglement: str = "circular",
        backend: str | None = None,
        shots: int = 1024,
    ) -> None:
        """
        Parameters
        ----------
        n_qubits : int
            Number of qubits in the circuit.
        n_layers : int
            Number of variational layers.
        entanglement : str
            Entanglement pattern ("circular", "full", "none").
        backend : str | None
            Name of the Pennylane device. If None, defaults to "default.qubit".
        shots : int
            Number of shots for expectation estimation.
        """
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement = entanglement
        self.shots = shots
        dev_name = backend or "default.qubit"
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)

        # Number of trainable parameters: each layer has n_qubits * 3 parameters
        self.n_params = n_layers * n_qubits * 3

        # Build the QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="numpy")

    def _circuit(self, *params):
        """Variational circuit."""
        for layer in range(self.n_layers):
            for q in range(self.n_qubits):
                idx = layer * self.n_qubits * 3 + q * 3
                qml.RX(params[idx + 0], wires=q)
                qml.RY(params[idx + 1], wires=q)
                qml.RZ(params[idx + 2], wires=q)
            if self.entanglement == "circular":
                for q in range(self.n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
            elif self.entanglement == "full":
                for q1 in range(self.n_qubits):
                    for q2 in range(q1 + 1, self.n_qubits):
                        qml.CNOT(wires=[q1, q2])
        return qml.expval(qml.PauliZ(0))

    def run(self, thetas: Iterable[float] | np.ndarray) -> np.ndarray:
        """
        Evaluate the circuit for a batch of parameter sets.

        Parameters
        ----------
        thetas : Iterable[float] | np.ndarray
            Array of shape (batch_size, n_params).

        Returns
        -------
        np.ndarray
            Expectation values of shape (batch_size, 1).
        """
        if isinstance(thetas, list):
            thetas = np.array(thetas, dtype=np.float32)
        if thetas.ndim == 1:
            thetas = thetas.reshape(1, -1)
        batch_size = thetas.shape[0]
        expectations = np.zeros((batch_size, 1), dtype=np.float64)

        for i in range(batch_size):
            params = thetas[i]
            if params.shape[0]!= self.n_params:
                raise ValueError(
                    f"Expected {self.n_params} parameters, got {params.shape[0]}"
                )
            expectation = self.qnode(*params)
            expectations[i] = expectation
        return expectations
