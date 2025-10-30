"""Quantum self‑attention using a Pennylane variational circuit."""

import pennylane as qml
import numpy as np
import torch

class SelfAttention:
    """
    Variational quantum self‑attention block.
    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be even, half encode inputs, half encode keys/values).
    n_layers : int, default 2
        Number of variational layers.
    """
    def __init__(self, n_qubits: int, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self._build_circuit()

    def _variational_layer(self, params, wires):
        for i, wire in enumerate(wires):
            qml.RX(params[0, i], wire)
            qml.RY(params[1, i], wire)
            qml.RZ(params[2, i], wire)
        for i in range(len(wires) - 1):
            qml.CNOT(wires[i], wires[i + 1])

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="torch")
        def circuit(rotation_params, entangle_params, inputs):
            # Encode classical inputs on first half of qubits
            for i in range(self.n_qubits // 2):
                qml.RX(inputs[:, i], i)
                qml.RY(inputs[:, i], i + 1)

            # Variational layers
            for l in range(self.n_layers):
                self._variational_layer(rotation_params[l], range(self.n_qubits))

            # Expectation values as attention weights
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        self.circuit = circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the variational circuit and return expectation values.
        Parameters
        ----------
        rotation_params : ndarray of shape (n_layers, 3, n_qubits)
        entangle_params : ndarray of shape (n_layers, n_qubits - 1)  # unused, retained for API
        inputs : ndarray of shape (batch, n_qubits // 2)
        shots : int, default 1024  # unused in the default simulator
        Returns
        -------
        ndarray of shape (batch, n_qubits) containing expectation values.
        """
        rotation_params_t = torch.tensor(rotation_params, dtype=torch.float32)
        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        result = self.circuit(rotation_params_t, entangle_params, inputs_t)
        return result.detach().numpy()

__all__ = ["SelfAttention"]
