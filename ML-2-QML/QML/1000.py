"""Quantum self‑attention using Pennylane variational circuits.

The circuit implements a parameterised rotation layer followed by a
controlled‑rotation entangling layer. The module exposes the same
``run`` method as the classical counterpart, returning a probability
distribution over the computational basis.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class SelfAttention:
    """Variational self‑attention circuit.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    num_layers : int, default=2
        Number of rotation–entanglement layers.
    dev_name : str, default="default.qubit"
        Pennylane device name.
    """

    def __init__(self, n_qubits: int, num_layers: int = 2, dev_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.num_layers = num_layers
        self.dev = qml.device(dev_name, wires=n_qubits)

    def _build_circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Return a Pennylane QNode that applies the parameterised circuit."""
        @qml.qnode(self.dev)
        def circuit():
            # Rotation layer
            for i in range(self.n_qubits):
                idx = 3 * i
                qml.RX(rotation_params[idx], wires=i)
                qml.RY(rotation_params[idx + 1], wires=i)
                qml.RZ(rotation_params[idx + 2], wires=i)

            # Entangling layer
            for i in range(self.n_qubits - 1):
                qml.CRX(entangle_params[i], wires=[i, i + 1])

            # Measure in computational basis
            return qml.probs(wires=range(self.n_qubits))

        return circuit

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        shots: int = 1024,
    ) -> dict:
        """Execute the circuit on the chosen backend.

        Parameters
        ----------
        rotation_params : np.ndarray
            Rotation angles of shape ``(3 * n_qubits,)``.
        entangle_params : np.ndarray
            Entanglement angles of shape ``(n_qubits - 1,)``.
        shots : int, default=1024
            Number of measurement shots.

        Returns
        -------
        dict
            Mapping from bitstring to probability.
        """
        circuit = self._build_circuit(rotation_params, entangle_params)
        probs = circuit()
        # Convert probabilities to a dictionary of bitstrings
        bitstrings = [format(i, f"0{self.n_qubits}b") for i in range(2 ** self.n_qubits)]
        return dict(zip(bitstrings, probs))

__all__ = ["SelfAttention"]
