"""Quantum multi‑head self‑attention using Pennylane.

The circuit consists of a layer of parameterised rotations followed by a
layer of controlled‑X entanglement.  Expectation values of the Pauli‑Z
operator are interpreted as attention logits and normalised with a
soft‑max.  The implementation is fully differentiable and can be
optimised with standard gradient‑based optimisers.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml

class SelfAttention:
    """
    Variational quantum circuit that emulates a single‑head self‑attention
    block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits used to encode one attention head.
    device : str | pennylane.Device, default "default.qubit"
        PennyLane device on which the circuit is executed.
    """

    def __init__(
        self,
        n_qubits: int = 4,
        device: str | qml.Device = "default.qubit",
    ) -> None:
        self.n_qubits = n_qubits
        if isinstance(device, str):
            self.dev = qml.device(device, wires=n_qubits)
        else:
            self.dev = device

    def _circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ) -> list[float]:
        """Return expectation values of Pauli‑Z for each qubit."""

        @qml.qnode(self.dev, interface="autograd")
        def circuit():
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit()

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational circuit and return attention logits.

        Parameters
        ----------
        rotation_params : np.ndarray, shape (3 * n_qubits,)
            Rotation angles for the RX/RY/RZ gates.
        entangle_params : np.ndarray, shape (n_qubits - 1,)
            Angles for the CRX entangling gates.
        shots : int
            Number of shots for sampling; ignored when using statevector.

        Returns
        -------
        np.ndarray
            Shape (n_qubits,) – soft‑maxed expectation values of Z.
        """
        # Obtain expectation values of Pauli‑Z
        logits = np.array(self._circuit(rotation_params, entangle_params))
        # Numerical stability for soft‑max
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return probs

__all__ = ["SelfAttention"]
