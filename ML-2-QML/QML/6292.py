import numpy as np
import pennylane as qml
from typing import Iterable

class FCL:
    """
    Variational quantum circuit implementing a fully connected layer.
    Supports multi‑qubit circuits, parameter‑shift gradient, and a hybrid
    loss function that can be used with classical targets.
    """
    def __init__(self, n_qubits: int = 2, device: str = "default.qubit", shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.device = qml.device(device, wires=n_qubits, shots=shots)
        self._build_qnode()

    def _build_qnode(self) -> None:
        @qml.qnode(self.device, interface="autograd")
        def circuit(params):
            # Apply a layer of H and parameterized Ry gates
            for i in range(self.n_qubits):
                qml.Hadamard(wires=i)
                qml.RY(params[i], wires=i)
            # Entangle with a simple CNOT chain
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Measure expectation of PauliZ on first qubit
            return qml.expval(qml.PauliZ(0))
        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Evaluate the circuit for a list of parameters.
        Returns a NumPy array with a single expectation value.
        """
        params = np.array(list(thetas), dtype=np.float32)
        if params.shape[0]!= self.n_qubits:
            raise ValueError(f"Parameter vector must have length {self.n_qubits}")
        expectation = self.circuit(params)
        return np.array([expectation])

    def grad(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation value w.r.t. each parameter
        using the parameter‑shift rule. Returns a NumPy array of gradients.
        """
        params = np.array(list(thetas), dtype=np.float32)
        if params.shape[0]!= self.n_qubits:
            raise ValueError(f"Parameter vector must have length {self.n_qubits}")
        shift = np.pi / 2
        grads = []
        for idx in range(self.n_qubits):
            shifted_plus = params.copy()
            shifted_minus = params.copy()
            shifted_plus[idx] += shift
            shifted_minus[idx] -= shift
            f_plus = self.circuit(shifted_plus)
            f_minus = self.circuit(shifted_minus)
            grads.append((f_plus - f_minus) / 2)
        return np.array(grads)

    def hybrid_loss(self, thetas: Iterable[float], target: float) -> float:
        """
        Simple mean‑squared error loss between the circuit expectation
        and a classical target value.
        """
        expectation = self.run(thetas)[0]
        return (expectation - target) ** 2
