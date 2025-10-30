"""Variational implementation of a fully connected layer.

The circuit encodes an input vector into a set of qubits, applies a
parameterised rotation layer that can be trained, and measures a single
observable.  The architecture mirrors the classical MLP above and can
be trained with the same loss function.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

class QuantumFullyConnectedLayer:
    """Quantum variational layer that emulates a small MLP.

    Parameters
    ----------
    n_qubits : int
        Number of qubits (must be >= 1).  One qubit per input feature.
    n_layers : int, optional
        Number of variational layers.
    device : str, optional
        Pennylane device name (default ``'default.qubit'``).
    shots : int, optional
        Number of shots for expectation estimation.
    """
    def __init__(self,
                 n_qubits: int = 1,
                 n_layers: int = 2,
                 device: str = "default.qubit",
                 shots: int = 1024) -> None:
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = qml.device(device, wires=n_qubits, shots=shots)
        self.params = pnp.random.randn(n_layers, n_qubits) * 0.1

        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs, params):
            # Data re‑uploading: encode inputs as Ry rotations
            for i, x in enumerate(inputs):
                qml.RY(x, wires=i)

            # Variational layers
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.RY(params[layer, qubit], wires=qubit)
                # Entangle with a simple chain of CNOTs
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])

            # Measure expectation of PauliZ on the first qubit
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Return the circuit output for each input row."""
        out = []
        for x in X:
            out.append(self.circuit(x, self.params))
        return np.array(out)

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            epochs: int = 200,
            lr: float = 1e-3,
            verbose: bool = False) -> None:
        """Gradient‑based training using the parameter‑shift rule."""
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        loss_fn = lambda preds, targets: ((preds - targets) ** 2).mean()

        for epoch in range(epochs):
            preds = self.forward(X)
            loss = loss_fn(preds, y)
            self.params = opt.step(loss, self.params)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs}  Loss: {loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for the given inputs."""
        return self.forward(X)

    def run(self, X: np.ndarray) -> np.ndarray:
        """Compatibility wrapper: ``run`` behaves like ``predict``."""
        return self.predict(X)

def FCL(n_qubits: int = 1,
        n_layers: int = 2,
        device: str = "default.qubit",
        shots: int = 1024):
    """Factory mirroring the original API."""
    return QuantumFullyConnectedLayer(n_qubits, n_layers, device, shots)

__all__ = ["FCL", "QuantumFullyConnectedLayer"]
