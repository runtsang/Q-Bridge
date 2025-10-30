"""Variational quantum circuit acting as a fully connected layer.

The implementation uses Pennylane to construct a parameterized circuit.
The `run` method evaluates the expectation value of Pauli‑Z on the first
qubit after applying a sequence of RX rotations and entangling CNOTs.
"""

from __future__ import annotations

from typing import Iterable, List

import pennylane as qml
import numpy as np


class FCL:
    """Quantum fully connected layer using a variational circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.  Must be >= 1.
    device_name : str
        Pennylane device name, e.g. 'default.qubit'.
    shots : int
        Number of shots for expectation estimation.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        device_name: str = "default.qubit",
        shots: int = 1000,
    ) -> None:
        self.n_qubits = n_qubits
        self.device = qml.device(device_name, wires=n_qubits, shots=shots)
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.device, interface="torch")
        def circuit(thetas: torch.Tensor) -> torch.Tensor:
            # Encode the input parameters as rotation angles on each qubit
            for i in range(self.n_qubits):
                qml.RX(thetas[i], wires=i)

            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Second layer of rotations
            for i in range(self.n_qubits):
                qml.RX(thetas[i + self.n_qubits], wires=i)

            # Measure expectation of Pauli‑Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit with a flat list of parameters.

        Parameters
        ----------
        thetas : Iterable[float]
            List of 2 * n_qubits parameters. The first n_qubits are used for
            the first rotation layer, the remaining for the second layer.

        Returns
        -------
        np.ndarray
            Expectation value of the first qubit as a single‑element array.
        """
        import torch

        # Ensure the list has the correct length
        if len(thetas)!= 2 * self.n_qubits:
            raise ValueError(
                f"Theta list must contain {2 * self.n_qubits} elements."
            )

        theta_tensor = torch.tensor(
            list(thetas), dtype=torch.float32
        )
        expectation = self.circuit(theta_tensor)
        return np.array([expectation.item()])

__all__ = ["FCL"]
