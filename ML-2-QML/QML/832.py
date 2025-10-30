"""Variational quantum circuit emulating a fully‑connected layer.

The original example used a single‑qubit circuit with a single parameter.
This version supports an arbitrary number of qubits, entanglement and
provides a ``run`` method that returns the expectation value of the Pauli‑Z
operator on the first qubit.  The circuit is implemented with Pennylane
and can be executed on simulators or real devices.
"""

import pennylane as qml
import torch
import numpy as np
from typing import Iterable

class FCL:
    """Parameterised quantum circuit mimicking a fully‑connected layer."""

    def __init__(
        self,
        n_qubits: int = 1,
        device: str = "default.qubit",
        shots: int = 1000,
    ) -> None:
        self.n_qubits = n_qubits
        self.device = qml.device(device, wires=n_qubits, shots=shots)
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.device, interface="torch")
        def circuit(thetas: torch.Tensor):
            # Prepare an entangled state
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            # Parameterised rotations
            for i, theta in enumerate(thetas):
                qml.RY(theta, wires=i)
            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measurement expectation
            return qml.expval(qml.PauliZ(0))
        return circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit and return the expectation value."""
        thetas = torch.as_tensor(list(thetas), dtype=torch.float32)
        expectation = self._circuit(thetas)
        return expectation.detach().numpy()

__all__ = ["FCL"]
