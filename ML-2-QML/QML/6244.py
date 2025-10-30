import pennylane as qml
import numpy as np
from typing import Iterable, List

class FCL:
    """
    Quantum implementation of a fully‑connected layer.  The circuit is
    parameterised by a list of rotation angles that are applied to each
    qubit.  The expectation value of the Pauli‑Z observable on the first
    qubit is returned as a 1‑D NumPy array.  The class offers a
    ``run`` method for forward evaluation, a ``gradient`` method that
    returns the parameter‑shift gradient, and a ``train`` helper that
    performs a few steps of stochastic gradient descent.
    """
    def __init__(self,
                 n_qubits: int = 1,
                 device_name: str = "default.qubit",
                 shots: int = 1,
                 seed: int | None = None) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits, shots=shots, seed=seed)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: np.ndarray) -> np.ndarray:
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def run(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Forward evaluation.  ``thetas`` must be a list of length
        ``n_qubits``.  The expectation value of Z on the first qubit is
        returned wrapped in a 1‑D NumPy array for API compatibility.
        """
        params = np.array(thetas, dtype=np.float64)
        if params.shape[0]!= self.n_qubits:
            raise ValueError(f"Expected {self.n_qubits} parameters, got {params.shape[0]}")
        val = self.circuit(params)
        return np.array([val])

    def gradient(self, thetas: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation value with respect to the
        parameters using the parameter‑shift rule.  The result is a 1‑D
        array of the same length as ``thetas``.
        """
        params = np.array(thetas, dtype=np.float64)
        grad = qml.grad(self.circuit)(params)
        return grad

    def train(self,
              init_thetas: Iterable[float],
              targets: Iterable[float],
              lr: float = 0.01,
              epochs: int = 50) -> List[float]:
        """
        Very small training routine that optimises the circuit parameters
        to minimise the mean‑squared‑error between the circuit outputs
        and ``targets``.  Returns the loss history.
        """
        params = np.array(init_thetas, dtype=np.float64)
        history = []
        for epoch in range(epochs):
            preds = self.run(params)
            loss = np.mean((preds - np.array(targets)) ** 2)
            grads = self.gradient(params)
            params -= lr * grads
            history.append(loss)
        return history
