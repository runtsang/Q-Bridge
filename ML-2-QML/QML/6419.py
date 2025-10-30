import pennylane as qml
import numpy as np

class SelfAttention:
    """
    Variational self‑attention circuit built with PennyLane.
    The interface matches the original seed: ``run(backend, rotation_params,
    entangle_params, shots)``; the angle arrays are ignored to preserve
    compatibility while the circuit learns its own parameters.
    """
    def __init__(self, n_qubits: int = 4):
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Initialize trainable parameters
        self.rotation_params = np.random.uniform(0, 2 * np.pi,
                                                 size=(n_qubits, 3))
        self.entangle_params = np.random.uniform(0, 2 * np.pi,
                                                 size=(n_qubits - 1,))
        self.params = np.concatenate([self.rotation_params.flatten(),
                                      self.entangle_params])

    def _circuit(self, params):
        rotation_params = params[:self.n_qubits * 3].reshape(self.n_qubits, 3)
        entangle_params = params[self.n_qubits * 3:].reshape(self.n_qubits - 1)

        for i in range(self.n_qubits):
            qml.RX(rotation_params[i, 0], wires=i)
            qml.RY(rotation_params[i, 1], wires=i)
            qml.RZ(rotation_params[i, 2], wires=i)

        for i in range(self.n_qubits - 1):
            qml.CRX(entangle_params[i], wires=[i, i + 1])

        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def run(self, backend, rotation_params: np.ndarray,
            entangle_params: np.ndarray, shots: int = 1024):
        """
        Execute the variational circuit on the supplied backend.
        Parameters are ignored to keep API compatibility.
        Returns expectation values of PauliZ on each qubit,
        interpreted as attention scores.
        """
        @qml.qnode(self.dev, interface="numpy")
        def circuit():
            return self._circuit(self.params)

        # If a custom backend is provided, replace the device
        if backend is not None and backend!= self.dev:
            self.dev = backend
            @qml.qnode(self.dev, interface="numpy")
            def circuit():
                return self._circuit(self.params)

        return circuit()

    def train(self, epochs: int = 200, lr: float = 0.01):
        """
        Train the circuit parameters using parameter‑shift gradients.
        """
        opt = qml.GradientDescentOptimizer(stepsize=lr)
        for _ in range(epochs):
            self.params = opt.step(self.loss, self.params)

    def loss(self, params):
        """
        Dummy loss that encourages uniform attention.
        """
        attn = self._circuit(params)
        return np.mean((attn - 1.0 / self.n_qubits) ** 2)
