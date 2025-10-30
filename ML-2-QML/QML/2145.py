"""Quantum self‑attention via a variational Pennylane circuit."""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp


class QuantumSelfAttention:
    """
    Variational quantum circuit that emulates a self‑attention block.
    Each qubit receives a rotation gate parameterized by ``rotation_params``.
    Adjacent qubits are entangled with CRX gates parameterized by ``entangle_params``.
    The expectation value of Pauli‑Z on each qubit is interpreted as an attention score.
    """

    def __init__(
        self,
        n_qubits: int,
        num_heads: int = 1,
        device_name: str = "default.qubit",
        device_shots: int = 1024,
    ):
        self.n_qubits = n_qubits
        self.num_heads = num_heads
        self.device = qml.device(device_name, wires=n_qubits, shots=device_shots)

    def _build_circuit(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ):
        @qml.qnode(self.device, interface="autograd")
        def circuit():
            # Apply rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)

            # Entanglement
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Return expectation values of Z as attention logits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int | None = None,
    ):
        """
        Executes the circuit and returns a dictionary of expectation values,
        one per qubit. The dictionary keys are qubit indices.
        """
        if shots is not None:
            self.device.shots = shots
        circuit = self._build_circuit(rotation_params, entangle_params)
        return {i: val for i, val in enumerate(circuit())}

    def gradient(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
    ):
        """
        Computes the gradient of the expectation values with respect to
        all circuit parameters using the parameter‑shift rule.
        Returns a dictionary mapping parameter indices to gradients.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        return qml.grad(circuit)(rotation_params, entangle_params)


def SelfAttention():
    """
    Factory that returns a QuantumSelfAttention instance configured for 4 qubits.
    """
    return QuantumSelfAttention(n_qubits=4)


__all__ = ["SelfAttention"]
