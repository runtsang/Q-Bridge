import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from typing import Iterable, List

class FCL:
    """
    Variational quantum circuit implementing a fully‑connected layer.
    The circuit uses H + Ry rotations followed by a chain of CNOTs.
    The `run` method returns the expectation value of Z on the first qubit.
    A `train_step` routine demonstrates gradient‑based optimisation.
    """

    def __init__(self, n_qubits: int = 1, dev: qml.Device = None):
        self.n_qubits = n_qubits
        self.dev = dev or qml.device("default.qubit", wires=n_qubits, shots=1024)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(thetas):
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(thetas[i], wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """Evaluate the circuit with the supplied theta vector."""
        theta_arr = pnp.array(list(thetas), dtype=pnp.float64)
        expectation = self.circuit(theta_arr)
        return np.array([expectation])

    def train_step(self, thetas: Iterable[float], target: float, lr: float = 0.1):
        """Perform one gradient‑descent step on the circuit parameters."""
        theta_arr = pnp.array(list(thetas), dtype=pnp.float64)
        loss = (self.circuit(theta_arr) - target) ** 2
        grads = qml.gradients.gradients(loss, [theta_arr])[0]
        theta_arr -= lr * grads
        return loss.item(), theta_arr
