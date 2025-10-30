"""Variational quantum filter for ConvGen092."""

import pennylane as qml
import numpy as np

class ConvGen092:
    """
    Variational quantum circuit that acts on a flattened feature vector
    of length ``kernel_size**2 * out_channels``. The circuit consists of
    a layer of RY rotations, a chain of CNOTs, and a final layer of
    parameterised RY rotations. The expectation value of PauliZ on
    qubit 0 is returned as the filter output.
    """

    def __init__(
        self,
        *,  # keyword‑only arguments
        kernel_size: int = 2,
        out_channels: int = 1,
        shots: int = 1024,
    ) -> None:
        self.n_qubits = kernel_size ** 2 * out_channels
        self.dev = qml.device("default.qubit", wires=self.n_qubits, shots=shots)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x, params):
            # Encode classical data with RY rotations
            for i in range(self.n_qubits):
                qml.RY(x[i], wires=i)
            # Entangle the qubits
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Apply variational parameters
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Observable
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        # Initialise parameters randomly
        self.params = np.random.randn(self.n_qubits)

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate the circuit on input vector ``x``.
        Parameters
        ----------
        x : np.ndarray
            1‑D array of length ``n_qubits``.
        Returns
        -------
        float
            Expectation value of PauliZ on qubit 0.
        """
        return float(self.circuit(x, self.params))
