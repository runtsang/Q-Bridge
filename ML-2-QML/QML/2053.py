"""Quantum fully‑connected layer implemented with PennyLane.

The circuit prepares an n‑qubit state, applies a layer of RX rotations
parameterised by an array of angles, entangles with CNOTs, and
measures the expectation value of Pauli‑Z on each qubit. The run
method returns a NumPy array of expectation values.
"""

import pennylane as qml
import numpy as np

class FullyConnectedLayer:
    """Variational circuit mimicking a fully‑connected layer."""
    def __init__(self, n_qubits: int, dev_name: str = "default.qubit", shots: int = 1024):
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # Prepare initial H state
            for w in range(n_qubits):
                qml.Hadamard(wires=w)
            # Parameterised RX rotation per qubit
            for w, theta in enumerate(params):
                qml.RX(theta, wires=w)
            # Entanglement layer (CNOT chain)
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            # Measure Z expectation on each qubit
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        self.circuit = circuit

    def run(self, thetas: np.ndarray) -> np.ndarray:
        """
        Execute the circuit with the supplied parameter array.
        Thetas must be a 1‑D array of length n_qubits.
        Returns a NumPy array of Z expectation values.
        """
        return np.array(self.circuit(thetas))
