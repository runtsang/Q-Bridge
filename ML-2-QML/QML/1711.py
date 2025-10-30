import numpy as np
import pennylane as qml
from typing import Iterable

class FullyConnectedLayer:
    """
    Quantum fully‑connected layer implemented with a PennyLane variational circuit.
    Uses RY rotations to encode each input value and measures the PauliZ expectation
    on the first qubit. Supports analytic gradients and batch evaluation.
    """

    def __init__(self, n_qubits: int = 1, dev: str = "default.qubit") -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(dev, wires=n_qubits)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(theta, *params):
            # Encode each input into a rotation about the Y axis.
            for w, t in zip(range(n_qubits), theta):
                qml.RY(t, wires=w)
            # Optional entanglement for multi‑qubit circuits.
            if n_qubits > 1:
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit for a batch of input parameters."""
        theta_arr = np.asarray(thetas, dtype=np.float64)
        if theta_arr.ndim == 1:
            theta_arr = theta_arr.reshape(1, -1)
        outputs = np.array([self.circuit(t) for t in theta_arr])
        return outputs.reshape(-1)

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """Analytic gradient of the expectation value w.r.t. the inputs."""
        theta_arr = np.asarray(thetas, dtype=np.float64)
        if theta_arr.ndim == 1:
            theta_arr = theta_arr.reshape(1, -1)
        grads = np.array([qml.grad(self.circuit)(t) for t in theta_arr])
        return grads.reshape(-1)

    def set_backend(self, dev: str) -> None:
        """Switch the underlying PennyLane device (e.g. 'default.qubit', 'qiskit.aer')."""
        self.dev = qml.device(dev, wires=self.n_qubits)
        # Re‑create the circuit on the new device.
        @qml.qnode(self.dev, interface="autograd")
        def circuit(theta, *params):
            for w, t in zip(range(self.n_qubits), theta):
                qml.RY(t, wires=w)
            if self.n_qubits > 1:
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

__all__ = ["FullyConnectedLayer"]
