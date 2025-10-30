"""Quantum neural network implementation of EstimatorQNN.

This module provides a variational circuit that accepts both input
features and trainable weights.  The circuit is built on Pennylane
and can be executed on any of its back‑ends (state‑vector, qasm, etc.).
The class exposes an :meth:`predict` method that returns the
expectation value of a chosen observable, mirroring the behaviour of
the classical counterpart while enabling hybrid training.
"""

import pennylane as qml
import torch
import numpy as np
from typing import Iterable, Sequence

class EstimatorQNN:
    """Variational quantum neural network for regression.

    The circuit encodes input features via RY rotations and applies a
    configurable number of variational layers consisting of RZ rotations
    and CNOT entanglement.  The observable is a single Pauli‑Y
    expectation value on the first qubit.  The implementation uses
    Pennylane's ``torch`` interface, allowing seamless hybrid training
    with PyTorch optimisers.
    """

    def __init__(
        self,
        num_qubits: int = 2,
        layers: int = 3,
        backend: str = "default.qubit",
    ) -> None:
        self.num_qubits = num_qubits
        self.layers = layers
        self.dev = qml.device(backend, wires=num_qubits)
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            """Return the expectation value of Pauli‑Y on qubit 0."""
            # Input encoding
            for i in range(self.num_qubits):
                qml.RY(inputs[i], wires=i)

            # Variational layers
            idx = 0
            for _ in range(self.layers):
                for i in range(self.num_qubits):
                    qml.RZ(weights[idx], wires=i)
                    idx += 1
                for i in range(self.num_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(self.num_qubits):
                    qml.RZ(weights[idx], wires=i)
                    idx += 1

            return qml.expval(qml.PauliY(0))

        self.qnode = circuit

    def predict(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Evaluate the circuit for a batch of inputs.

        Parameters
        ----------
        inputs : array‑like, shape (batch, num_qubits)
            Feature vectors.
        weights : array‑like, shape (layers * num_qubits * 2,)
            Trainable parameters.

        Returns
        -------
        outputs : np.ndarray, shape (batch,)
            Expectation values.
        """
        outputs = []
        for inp in inputs:
            inp_t = torch.tensor(inp, dtype=torch.float32)
            w_t = torch.tensor(weights, dtype=torch.float32)
            out = self.qnode(inp_t, w_t).detach().numpy()
            outputs.append(out)
        return np.array(outputs)

__all__ = ["EstimatorQNN"]
