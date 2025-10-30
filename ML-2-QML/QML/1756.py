"""Quantum self‑attention using Pennylane variational circuit.

The module defines a `SelfAttention` class that builds a
parameter‑shaped circuit with single‑qubit rotations and controlled‑RZ
entanglement.  The run method executes the circuit on a Pennylane
device and returns a probability distribution over the computational
basis, which can be interpreted as attention weights.  The interface
mirrors the classical version for seamless hybrid usage.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class SelfAttention:
    """
    Variational quantum self‑attention.

    Args:
        n_qubits (int): Number of qubits (must equal embed_dim / 3).
        device_name (str, optional): Pennylane device to use. Defaults to "default.qubit".
        shots (int, optional): Number of measurement shots. Defaults to 1024.
    """

    def __init__(self, n_qubits: int, device_name: str = "default.qubit", shots: int = 1024):
        self.n_qubits = n_qubits
        self.device = qml.device(device_name, wires=n_qubits, shots=shots)
        self.shots = shots

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        @qml.qnode(self.device)
        def circuit():
            # Rotations
            for i in range(self.n_qubits):
                qml.RX(rotation_params[3 * i], wires=i)
                qml.RY(rotation_params[3 * i + 1], wires=i)
                qml.RZ(rotation_params[3 * i + 2], wires=i)
            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])
            return qml.probs(wires=range(self.n_qubits))
        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Execute the variational circuit.

        Args:
            rotation_params (np.ndarray): Shape (3 * n_qubits,).
            entangle_params (np.ndarray): Shape (n_qubits - 1,).
            inputs (np.ndarray, optional): Not used in this simplified version
                but kept for API compatibility.

        Returns:
            np.ndarray: Probability distribution over 2**n_qubits basis states.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        probs = circuit()
        return probs


__all__ = ["SelfAttention"]
