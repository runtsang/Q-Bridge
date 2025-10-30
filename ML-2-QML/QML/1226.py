"""Quantum fully‑connected layer using a PennyLane variational circuit."""
import pennylane as qml
import numpy as np


def FCL() -> qml.QNode:
    """Return a variational circuit that mimics a fully‑connected layer."""
    class QuantumLayer:
        def __init__(
            self,
            n_qubits: int = 1,
            dev: qml.Device | None = None,
            shots: int = 1000,
        ) -> None:
            self.n_qubits = n_qubits
            self.dev = dev or qml.device("default.qubit", wires=n_qubits, shots=shots)
            self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

        def _circuit(self, *thetas: float) -> float:
            # H on all qubits
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
            # Parameterised rotations
            for i, theta in enumerate(thetas):
                qml.RY(theta, wires=i)
            # Optional entanglement
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure Pauli‑Z on the first qubit
            return qml.expval(qml.PauliZ(0))

        def run(self, thetas: Iterable[float]) -> np.ndarray:
            """Evaluate the circuit and return the expectation value."""
            expectation = self.qnode(*thetas)
            return np.array([expectation])

    return QuantumLayer()


__all__ = ["FCL"]
