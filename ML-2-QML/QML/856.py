"""Quantum fully connected layer using Pennylane variational circuit."""

import pennylane as qml
import numpy as np

class FCL:
    """
    Variational quantum circuit that emulates a fully connected layer.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the circuit.
    layers : int
        Number of variational layers.
    dev : pennylane.Device | None
        PennyLane quantum device. Defaults to the default qubit simulator.
    """

    def __init__(self, n_qubits: int = 1, layers: int = 1, dev: qml.Device | None = None) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            """Parameterized circuit with rotation layers and entanglement."""
            # Encode input parameters as rotation angles
            for i in range(self.n_qubits):
                qml.RY(params[0, i], wires=i)

            # Variational layers
            for l in range(self.layers):
                for i in range(self.n_qubits):
                    qml.RZ(params[l + 1, i], wires=i)
                # Entanglement
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                for i in range(self.n_qubits - 1, 0, -1):
                    qml.CNOT(wires=[i, i - 1])

            # Expectation value of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: np.ndarray | list[float]) -> np.ndarray:
        """
        Execute the variational circuit with the supplied parameters.

        Parameters
        ----------
        thetas : array-like
            Flattened array of parameters. Its shape must be
            (layers + 1, n_qubits) to match the circuit's expectations.

        Returns
        -------
        np.ndarray
            1‑D array containing the expectation value.
        """
        params = np.array(thetas, dtype=np.float32).reshape(self.layers + 1, self.n_qubits)
        expectation = self.circuit(params)
        return np.array([expectation])

__all__ = ["FCL"]
