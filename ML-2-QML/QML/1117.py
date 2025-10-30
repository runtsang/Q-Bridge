import pennylane as qml
import numpy as np
from typing import Iterable

class FCL:
    """
    Variational quantum circuit emulating a fully connected layer.
    Uses a parameterized ansatz with Ry rotations and CNOT entanglement.
    Returns the expectation value of PauliZ on the first qubit.
    """

    def __init__(self,
                 n_qubits: int = 4,
                 wires: Iterable[int] | None = None,
                 device: str = "default.qubit",
                 shots: int = 1024,
                 layers: int = 2) -> None:
        self.n_qubits = n_qubits
        self.wires = wires or list(range(n_qubits))
        self.shots = shots
        self.layers = layers
        self.dev = qml.device(device, wires=self.wires, shots=shots)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params, thetas):
            # Encode input features into Ry rotations
            for i, theta in enumerate(thetas):
                qml.RY(theta, wires=self.wires[i])

            # Variational layers
            for _ in range(self.layers):
                for i in range(self.n_qubits):
                    qml.RY(params[i], wires=self.wires[i])
                # Entangling pattern
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[self.wires[i], self.wires[i + 1]])
                qml.CNOT(wires=[self.wires[-1], self.wires[0]])

            # Measurement
            return qml.expval(qml.PauliZ(self.wires[0]))

        self.circuit = circuit

    def run(self, thetas: np.ndarray, params: np.ndarray) -> np.ndarray:
        """
        Compute the circuit expectation for each input sample.
        ``thetas``: 2‑D array (batch, n_qubits) of input angles.
        ``params``: 1‑D array of variational parameters of length n_qubits.
        Returns a 1‑D array of expectation values.
        """
        batch = []
        for theta in thetas:
            val = self.circuit(params, theta)
            batch.append(val)
        return np.array(batch)
