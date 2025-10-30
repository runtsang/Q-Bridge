"""Quantum self‑attention implemented with Pennylane.

The implementation uses a variational circuit that mimics the
classical attention workflow:

* Each qubit receives a basis‑encoding of the input vector.
* A trainable rotation layer (rx/ry/rz) implements the “query”
  transformation.
* A trainable entanglement layer (crx) implements the “key”
  transformation.
* The expectation values of the Pauli‑Z observables are returned as
  the attention output.

The circuit is wrapped in a Pennylane QuantumNode, so gradients can
be propagated through the parameters if desired.  The public API
matches the original `SelfAttention()` helper: a `run` method that
accepts a device, parameter arrays and a classical input vector.
"""

import numpy as np
import pennylane as qml
import torch

class SelfAttentionQuantum:
    """
    Variational quantum self‑attention block.

    Parameters
    ----------
    n_qubits : int
        Number of qubits / dimensionality of the input.
    num_layers : int, default 2
        Number of rotation/entanglement layers.
    """

    def __init__(self, n_qubits: int, num_layers: int = 2):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.wires = list(range(n_qubits))
        self.device = None
        self.qnode = None

    def _build_qnode(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ):
        """
        Construct a Pennylane QuantumNode that encodes the input
        and applies the variational layers.
        """
        @qml.qnode(self.device, interface="torch", diff_method="backprop")
        def circuit():
            # Basis encoding of the classical input
            for i, val in enumerate(inputs):
                qml.RX(val, wires=i)

            # Variational layers
            for _ in range(self.num_layers):
                for i in range(self.n_qubits):
                    idx = 3 * i
                    qml.RX(rotation_params[idx], wires=i)
                    qml.RY(rotation_params[idx + 1], wires=i)
                    qml.RZ(rotation_params[idx + 2], wires=i)
                for i in range(self.n_qubits - 1):
                    qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Return expectation values of Pauli‑Z
            return [qml.expval(qml.PauliZ(i)) for i in self.wires]

        return circuit

    def run(
        self,
        device: qml.Device,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
        shots: int = 1024,
    ) -> np.ndarray:
        """
        Execute the quantum attention circuit.

        Parameters
        ----------
        device : pennylane.Device
            Quantum device (e.g., default.qubit, qasm_simulator).
        rotation_params : np.ndarray
            Rotation parameters, shape (3 * n_qubits,).
        entangle_params : np.ndarray
            Entanglement parameters, shape (n_qubits - 1,).
        inputs : np.ndarray
            Classical input vector, shape (n_qubits,).
        shots : int, default 1024
            Number of shots for measurement.

        Returns
        -------
        np.ndarray
            Expectation values of Pauli‑Z observables as a NumPy array.
        """
        self.device = device
        self.device.shots = shots
        circuit = self._build_qnode(rotation_params, entangle_params, inputs)
        result = circuit()
        # Convert torch tensor to NumPy array
        return result.detach().cpu().numpy()

def SelfAttention():
    """
    Factory matching the original interface.

    Returns
    -------
    SelfAttentionQuantum
        Quantum self‑attention instance ready for use.
    """
    return SelfAttentionQuantum(n_qubits=4)

__all__ = ["SelfAttentionQuantum", "SelfAttention"]
