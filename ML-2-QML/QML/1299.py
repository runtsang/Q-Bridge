import numpy as np
import pennylane as qml
from typing import Iterable

class FCLayerQuantum:
    """
    Variational quantum circuit that emulates a fully‑connected layer.
    Provides a ``run`` method compatible with the original seed and
    exposes gradient computation via Pennylane’s autograd interface.
    """

    def __init__(self, n_qubits: int = 4, device: str = "default.qubit", shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits, shots=shots)
        self.n_params = n_qubits + 1  # one RY per qubit + global RY
        self._build_circuit()

    def _build_circuit(self) -> None:
        @qml.qnode(self.dev, interface="autograd")
        def circuit(params: np.ndarray) -> float:
            # Layer 1 – rotate each qubit
            for i in range(self.n_qubits):
                qml.RY(params[i], wires=i)
            # Layer 2 – entangle via CNOT chain
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            # Global rotation
            qml.RY(params[-1], wires=0)
            # Measurement expectation of Pauli‑Z on first qubit
            return qml.expval(qml.PauliZ(0))
        self._circuit = circuit

    def run(self, thetas: Iterable[Iterable[float]]) -> np.ndarray:
        """
        Execute the circuit for each parameter vector in ``thetas``.

        Parameters
        ----------
        thetas : Iterable[Iterable[float]]
            A sequence of parameter vectors, each of length ``n_params``.

        Returns
        -------
        numpy.ndarray
            Array of expectation values, one per parameter set.
        """
        expectations = []
        for theta in thetas:
            theta_arr = np.array(theta, dtype=np.float32)
            if theta_arr.size!= self.n_params:
                raise ValueError(
                    f"Expected {self.n_params} parameters per circuit, got {theta_arr.size}"
                )
            expectations.append(self._circuit(theta_arr))
        return np.array(expectations)

    def grad(self, theta: Iterable[float]) -> np.ndarray:
        """
        Compute the gradient of the expectation with respect to the parameters.

        Parameters
        ----------
        theta : Iterable[float]
            Parameter vector of length ``n_params``.

        Returns
        -------
        numpy.ndarray
            Gradient vector.
        """
        theta_arr = np.array(theta, dtype=np.float32)
        return qml.grad(self._circuit)(theta_arr)

    def train_on_synthetic(self, epochs: int = 200, lr: float = 0.01) -> None:
        """
        Simple training loop using the parameter‑shift rule to minimise a
        mean‑squared‑error loss on synthetic data.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        loss_fn = lambda y_pred, y_true: np.mean((y_pred - y_true) ** 2)
        for _ in range(epochs):
            # Random synthetic data
            theta = np.random.randn(self.n_params)
            y_true = np.random.randn()
            y_pred = self._circuit(theta)
            loss = loss_fn(y_pred, y_true)
            gradients = qml.grad(self._circuit)(theta)
            theta -= lr * gradients

__all__ = ["FCLayerQuantum"]
