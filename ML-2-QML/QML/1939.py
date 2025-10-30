"""Quantum fully connected layer implemented with PennyLane.

The circuit applies a single RY rotation to each qubit followed by a
chain of CNOTs to create entanglement.  The expectation value of the
Pauli‑Z operator on each wire is returned as the layer output.  This
provides a quantum analogue of a classical linear layer with a
non‑linear activation when combined with subsequent processing.
"""

import numpy as np
import pennylane as qml
from typing import Iterable, Sequence


class QuantumFullyConnectedLayer:
    """Parameterized quantum circuit that mimics a fully‑connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits / input features.  Each feature is encoded as an
        RY rotation on a distinct qubit.
    backend : str, default="default.qubit"
        PennyLane backend identifier.
    shots : int, default=1024
        Number of shots for expectation estimation.  When ``shots`` is
        ``None`` the device returns a noiseless expectation value.
    """

    def __init__(self, n_qubits: int, backend: str = "default.qubit", shots: int | None = 1024) -> None:
        self.n_qubits = n_qubits
        self.backend = backend
        self.shots = shots
        self.dev = qml.device(backend, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(thetas: Sequence[float]):
            # Encode parameters
            for i in range(n_qubits):
                qml.RY(thetas[i], wires=i)

            # Entangle qubits
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])

            # Return a list of Pauli‑Z expectation values
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = circuit

    def run(self, thetas: Iterable[Sequence[float]]) -> np.ndarray:
        """Evaluate the circuit for one or more parameter vectors.

        Parameters
        ----------
        thetas
            Either a single sequence of length ``n_qubits`` or an iterable of
            such sequences.  When a batch is supplied, each vector is
            evaluated independently.

        Returns
        -------
        np.ndarray
            If ``thetas`` is a single vector, the shape is ``(n_qubits,)``.
            For a batch the shape is ``(batch, n_qubits)``.
        """
        # Normalize input to a list of vectors
        if isinstance(thetas, (list, tuple, np.ndarray)):
            # Check if the first element is a scalar (single vector)
            if isinstance(thetas[0], (float, int)):
                theta_arr = np.array([thetas], dtype=float)
            else:
                theta_arr = np.array(thetas, dtype=float)
        else:
            raise TypeError("`thetas` must be an iterable of floats or a single sequence")

        # Evaluate each vector
        results = []
        for theta in theta_arr:
            if len(theta)!= self.n_qubits:
                raise ValueError(f"Expected {self.n_qubits} parameters, got {len(theta)}")
            results.append(np.array(self._circuit(theta)))

        return np.array(results)

    def parameters(self):
        """Return the underlying device parameters for introspection."""
        return self._circuit.parameters


__all__ = ["QuantumFullyConnectedLayer"]
