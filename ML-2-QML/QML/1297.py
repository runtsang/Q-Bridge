"""Variational quantum circuit emulating a fully connected layer.

Uses Pennylane to construct a parameterised circuit with entanglement.
The circuit implements a simple feed‑forward mapping from a single
parameter to a Pauli‑Z expectation value.  The class can be used as a
drop‑in replacement for the original Qiskit example while providing
more flexibility (multiple qubits, adjustable entanglement, shot
control).
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


class FCL:
    """
    Variational quantum circuit that mimics a fully connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    device : str, optional
        Pennylane device name (default 'default.qubit').
    shots : int, optional
        Number of shots for expectation estimation.
    entangle : bool, optional
        Whether to add a CNOT chain between qubits.
    """

    def __init__(
        self,
        n_qubits: int = 1,
        device: str = "default.qubit",
        shots: int = 1000,
        entangle: bool = True,
    ) -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device(device, wires=n_qubits, shots=shots)
        self.entangle = entangle
        self._circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # Apply a layer of H gates
            for w in range(self.n_qubits):
                qml.Hadamard(wires=w)

            # Parameterised Ry rotations
            for w, p in enumerate(params):
                qml.RY(p, wires=w)

            # Optional entanglement
            if self.entangle and self.n_qubits > 1:
                for w in range(self.n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])

            # Measure expectation of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))

        return circuit

    def run(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Evaluate the circuit for a batch of parameters.

        Parameters
        ----------
        thetas : array‑like, shape (batch, n_qubits)
            Parameter values for each qubit.

        Returns
        -------
        np.ndarray
            Expectation values for each set of parameters.
        """
        if isinstance(thetas, list):
            thetas = np.array(thetas)

        # Ensure shape (batch, n_qubits)
        thetas = np.atleast_2d(thetas)
        if thetas.shape[1]!= self.n_qubits:
            raise ValueError(
                f"Expected {self.n_qubits} parameters per sample, got {thetas.shape[1]}"
            )

        # Compute expectations in batch
        expectations = np.array([self._circuit(t) for t in thetas])
        return expectations.reshape(-1, 1)

    def __repr__(self) -> str:
        return f"FCL(n_qubits={self.n_qubits}, shots={self.shots}, entangle={self.entangle})"


__all__ = ["FCL"]
