import pennylane as qml
import numpy as np
from typing import Iterable

class HybridFCL:
    """
    Quantum‑parameterized fully‑connected layer implemented with Pennylane.
    The circuit consists of alternating rotations and a simple entangling
    pattern.  The number of qubits is inferred from the dimensionality of the
    input parameters.
    Parameters
    ----------
    n_qubits : int
        Number of qubits (and therefore the dimensionality of the parameter
        vector).
    device : str, optional
        Pennylane device name (default ``"default.qubit"``, can also be
        ``"qiskit.aer"`` or any other supported backend).
    shots : int, optional
        Number of shots for the expectation value evaluation.
    """

    def __init__(self, n_qubits: int, device: str = "default.qubit", shots: int = 1000):
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits, shots=shots)
        self.circuit = self._build_circuit()

    def _build_circuit(self):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params):
            # Apply a rotation to each qubit
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Simple ring entanglement
            for i in range(self.n_qubits):
                qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            # Return the expectation value of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))
        return circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for a list of parameters.  The length of *thetas*
        must match ``n_qubits``.  Returns a one‑dimensional NumPy array
        containing the expectation value.
        """
        params = np.array(thetas, dtype=np.float32)
        expectation = self.circuit(params)
        return np.array([expectation])

    def train_step(self, thetas: Iterable[float], loss_fn, lr: float = 0.01):
        """
        Perform a single gradient‑descent step using Pennylane’s autograd
        interface.  Returns the updated parameters and the loss value.
        """
        params = np.array(thetas, dtype=np.float32)
        grads = qml.grad(self.circuit)(params)
        updated = params - lr * grads
        loss = loss_fn(updated, np.array([0.0]))
        return updated, loss

__all__ = ["HybridFCL"]
