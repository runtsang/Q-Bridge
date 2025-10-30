import pennylane as qml
import numpy as np
from pennylane.optimize import AdamOptimizer

class EstimatorQNN:
    """
    Variational quantum neural network for regression.
    Parameters
    ----------
    n_qubits : int, default 4
        Number of qubits in the device.
    layers : int, default 2
        Number of variational layers.
    entanglement : str, default 'full'
        Entanglement pattern ('full' or 'circular').
    obs : pennylane operation, default PauliZ(0)
        Observable whose expectation value is returned.
    dev : pennylane.Device, optional
        Quantum device; if None a default.qubit device is used.
    """
    def __init__(self,
                 n_qubits: int = 4,
                 layers: int = 2,
                 entanglement: str = 'full',
                 obs: qml.operation.Operator = None,
                 dev: qml.Device = None) -> None:
        self.n_qubits = n_qubits
        self.layers = layers
        self.entanglement = entanglement
        self.dev = dev or qml.device("default.qubit", wires=n_qubits)
        self.obs = obs or qml.PauliZ(0)

        # Create the QNode
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

        # Initialise trainable parameters
        self.params = np.random.uniform(0, 2 * np.pi,
                                        size=(layers, n_qubits))

    def _circuit(self, x, params):
        """Encode classical input and execute variational layers."""
        # Input encoding
        for i in range(self.n_qubits):
            qml.RY(x[i], wires=i)

        # Variational layers
        for l in range(self.layers):
            for i in range(self.n_qubits):
                qml.RY(params[l, i], wires=i)
            if self.entanglement == 'full':
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])
            elif self.entanglement == 'circular':
                for i in range(self.n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])

        return qml.expval(self.obs)

    def loss(self, params, X, y):
        preds = np.array([self.qnode(x, params) for x in X])
        return np.mean((preds - y) ** 2)

    def fit(self,
            X,
            y,
            epochs: int = 200,
            lr: float = 0.01,
            verbose: bool = False) -> None:
        """
        Train the variational circuit.
        Parameters
        ----------
        X : array‑like, shape (n_samples, n_qubits)
            Input features must match the number of qubits.
        y : array‑like, shape (n_samples, )
        """
        opt = AdamOptimizer(lr)
        params = self.params
        for epoch in range(epochs):
            params, loss_val = opt.step_and_cost(lambda p: self.loss(p, X, y), params)
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {loss_val:.6f}")
        self.params = params

    def predict(self, X):
        """
        Predict on new data.
        Parameters
        ----------
        X : array‑like, shape (n_samples, n_qubits)
        Returns
        -------
        preds : np.ndarray, shape (n_samples, )
        """
        return np.array([self.qnode(x, self.params) for x in X])

__all__ = ["EstimatorQNN"]
