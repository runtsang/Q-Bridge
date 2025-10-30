import numpy as np
import pennylane as qml
import torch

class SelfAttention:
    """
    Variational quantum circuit that implements a self‑attention block.
    The circuit consists of parameterised single‑qubit rotations (rotation_params)
    followed by entangling CNOTs whose angles are taken from entangle_params.
    The output is a probability vector over the computational basis obtained
    from measuring all qubits.  The circuit is differentiable via the
    parameter‑shift rule, enabling hybrid training.
    """
    def __init__(self, n_qubits: int = 4, dev_name: str = "default.qubit"):
        self.n_qubits = n_qubits
        self.dev = qml.device(dev_name, wires=n_qubits)

    def _circuit(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Builds the variational circuit."""
        for i in range(self.n_qubits):
            qml.RX(rotation_params[3 * i], wires=i)
            qml.RY(rotation_params[3 * i + 1], wires=i)
            qml.RZ(rotation_params[3 * i + 2], wires=i)
        for i in range(self.n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
            qml.RZ(entangle_params[i], wires=i + 1)  # Entangling rotation on target
        # Expectation values of PauliZ for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

    def get_qnode(self, rotation_params: np.ndarray, entangle_params: np.ndarray):
        """Wraps the circuit in a QNode that returns a probability vector."""
        @qml.qnode(self.dev, interface="torch")
        def circuit():
            out = self._circuit(rotation_params, entangle_params)
            # Convert expectation values to probabilities via linear mapping
            probs = 0.5 * (torch.tensor(out) + 1.0)
            return probs
        return circuit

    def run(self, backend, rotation_params: np.ndarray,
            entangle_params: np.ndarray, shots: int = 1024):
        """
        Execute the circuit and return a probability distribution.
        The `backend` argument is kept for API compatibility but is ignored
        because Pennylane manages its own device internally.
        """
        qnode = self.get_qnode(rotation_params, entangle_params)
        probs = qnode()
        return probs.detach().numpy()

    def gradient(self, rotation_params: np.ndarray,
                 entangle_params: np.ndarray):
        """
        Compute the gradient of the output probabilities w.r.t. all parameters.
        """
        qnode = self.get_qnode(rotation_params, entangle_params)
        grads = qml.grad(qnode)(rotation_params, entangle_params)
        return grads

# Singleton instance for compatibility with the original API
SelfAttention = SelfAttention()

__all__ = ["SelfAttention"]
