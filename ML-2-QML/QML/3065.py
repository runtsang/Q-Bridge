"""Quantum component of the hybrid fraud‑detection model.

This module implements a simple parameterised quantum circuit that
produces two expectation values, one per qubit.  It is designed to
serve as a feature extractor for the classical network.

The circuit uses a single RX rotation on each qubit followed by an
entangling RZZ gate.  The parameters of the rotations come from the
input data; the entanglement strength is fixed at zero for
demonstration purposes but can be learned if desired.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

class QuantumFraudLayer:
    """
    2‑qubit PennyLane qnode that returns expectation values of Pauli‑Z
    on each qubit.

    Parameters
    ----------
    device : str, optional
        PennyLane device name (default: ``"default.qubit"``).
    shots : int, optional
        Number of shots for the simulation (default: 1024).

    Methods
    -------
    forward(inputs : np.ndarray) -> np.ndarray
        Map a 2‑dimensional input vector to a 2‑dimensional output
        vector of expectation values.
    """

    def __init__(self, device: str = "default.qubit", shots: int = 1024) -> None:
        self.device = device
        self.shots = shots
        self.qnode = qml.QNode(self._circuit, qml.device(device, wires=2, shots=shots))

    def _circuit(self, r1: float, r2: float, entangle_strength: float) -> list[float]:
        qml.RX(r1, wires=0)
        qml.RX(r2, wires=1)
        qml.RZZ(entangle_strength, wires=[0, 1])
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        inputs : np.ndarray of shape (batch, 2)
            Each row contains the parameters (r1, r2) for the quantum circuit.
            The entanglement strength is fixed at 0.0 for simplicity.

        Returns
        -------
        np.ndarray of shape (batch, 2)
            Expectation values from the two qubits.
        """
        batch = inputs.astype(np.float32)
        outputs = []
        for r1, r2 in batch:
            expvals = self.qnode(r1, r2, 0.0)
            outputs.append(expvals)
        return np.array(outputs, dtype=np.float32)

__all__ = ["QuantumFraudLayer"]
