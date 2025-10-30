"""Quantum convolutional filter using a parameterized variational circuit.

The interface mirrors the classical ConvHybrid to allow drop‑in replacement.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

__all__ = ["ConvHybrid"]


class ConvHybrid:
    """
    Quantum implementation of a convolutional filter.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the convolution kernel (defines number of qubits).
    threshold : float, default 0.0
        Threshold applied to classical data before encoding.
    device : str, default "default.qubit"
        Pennylane device name.
    shots : int, default 1000
        Number of shots for measurement.
    """

    def __init__(
        self,
        kernel_size: int = 2,
        threshold: float = 0.0,
        device: str = "default.qubit",
        shots: int = 1000,
    ) -> None:
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.n_qubits = kernel_size ** 2
        self.device = qml.device(device, wires=self.n_qubits, shots=shots)
        self.params = pnp.random.uniform(0, 2 * np.pi, self.n_qubits)

        @qml.qnode(self.device)
        def circuit(x, params):
            for i, val in enumerate(x):
                qml.RX(np.pi if val > self.threshold else 0.0, wires=i)
            for i, p in enumerate(params):
                qml.RY(p, wires=i)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        self.circuit = circuit

    def run(self, data: np.ndarray) -> float:
        """Run the variational circuit on a single data patch."""
        data = np.reshape(data, (-1, self.n_qubits))
        probs = []
        for dat in data:
            expvals = self.circuit(dat, self.params)
            probs.append((1 - np.array(expvals)) / 2.0)
        probs = np.mean(probs, axis=0)
        return probs.mean().item()

    def fit(
        self,
        train_loader,
        epochs: int = 5,
        lr: float = 0.01,
    ) -> None:
        """Simple training loop using parameter‑shift rule."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            for x, y in train_loader:
                def loss_fn(params):
                    preds = []
                    for dat in x:
                        expvals = self.circuit(dat, params)
                        probs = (1 - np.array(expvals)) / 2.0
                        preds.append(probs.mean())
                    preds = np.array(preds)
                    return np.mean((preds - y) ** 2)

                self.params = opt.step(loss_fn, self.params)
