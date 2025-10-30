import pennylane as qml
import numpy as np
from typing import Iterable

class FullyConnectedLayer:
    """Quantum fully‑connected layer using a variational circuit.

    The circuit consists of a layer of Ry rotations followed by a chain
    of CNOTs that entangles all qubits. The expectation value of the
    Pauli‑Z operator on the first qubit is returned. The class exposes
    a ``run`` method that accepts an iterable of angles and returns the
    expectation as a NumPy array, enabling direct comparison with the
    classical implementation.
    """

    def __init__(self, n_qubits: int = 1, shots: int = 1000, device: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.shots = shots
        self.dev = qml.device(device, wires=n_qubits, shots=shots)

        @qml.qnode(self.dev)
        def circuit(theta):
            # Parameterized Ry rotations
            for w, t in enumerate(theta):
                qml.RY(t, wires=w)
            # Entanglement layer (chain of CNOTs)
            for w in range(n_qubits - 1):
                qml.CNOT(wires=[w, w + 1])
            # Measure expectation of Z on the first qubit
            return qml.expval(qml.PauliZ(0))
        self._circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit for each theta in the iterable."""
        expectations = []
        for theta in thetas:
            # Pad theta list to match the number of qubits
            theta_vec = [theta] + [0.0] * (self.n_qubits - 1)
            expectations.append(self._circuit(theta_vec))
        return np.array(expectations)
