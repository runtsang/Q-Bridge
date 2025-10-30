"""HybridFCL – a Pennylane variational circuit acting as a quantum fully connected layer.

The quantum module mirrors the classical architecture by providing a
``run`` method that accepts a list of parameters, executes a
parameterised variational circuit on a qubit device, and returns the
expectation value of the Pauli‑Z observable on the first qubit.
"""

__all__ = ["HybridFCL"]

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import Iterable, Sequence


class HybridFCL:
    """Variational quantum circuit with configurable entanglement depth.

    Parameters
    ----------
    n_qubits : int
        Number of qubits; each qubit receives a rotation about the Y axis.
    entanglement_depth : int, optional
        Number of nearest‑neighbour entangling layers (default 1).
    device : pennylane.Device, optional
        Pennylane device to use; defaults to the local qasm simulator.
    shots : int, optional
        Number of measurement shots; useful for simulating hardware noise.
    """

    def __init__(
        self,
        n_qubits: int,
        entanglement_depth: int = 1,
        device: qml.Device | None = None,
        shots: int = 1024,
    ) -> None:
        self.n_qubits = n_qubits
        self.entanglement_depth = entanglement_depth
        self.shots = shots

        self.dev = device or qml.device("default.qubit", wires=n_qubits, shots=shots)

        # Build a parameter‑shifting circuit
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: Sequence[float]) -> float:
            # Apply a rotation to each qubit
            for w in range(n_qubits):
                qml.RY(params[w], wires=w)

            # Entanglement layers
            for _ in range(entanglement_depth):
                for w in range(n_qubits - 1):
                    qml.CNOT(wires=[w, w + 1])

            # Measure Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Execute the circuit for a batch of parameter vectors.

        Parameters
        ----------
        thetas : Iterable[float]
            Iterable of parameter lists; each list must contain
            ``n_qubits`` angles.  The method returns a NumPy array of
            shape ``(1,)`` containing the mean expectation value across
            the batch, matching the signature of the original seed.

        Returns
        -------
        np.ndarray
            Array of shape ``(1,)`` with the mean expectation value.
        """
        values = np.asarray(list(thetas), dtype=np.float32)
        if values.ndim == 1:
            # Single sample
            expectation = self.circuit(values)
        else:
            # Batch: compute expectation for each sample
            expectations = np.array([self.circuit(v) for v in values])
            expectation = expectations.mean()

        return np.array([expectation])
