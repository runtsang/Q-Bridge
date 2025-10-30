import pennylane as qml
import numpy as np
import torch

class HybridSamplerQNN:
    """
    Quantum sampler network using a parameterized ansatz.
    Supports 2 qubits with trainable rotation angles and entanglement.
    """
    def __init__(self, n_qubits: int = 2, n_layers: int = 2, seed: int | None = None):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.params_shape = (n_layers, n_qubits, 3)  # RX, RZ, RY angles per qubit per layer

        @qml.qnode(self.dev, interface="torch")
        def circuit(inputs: torch.Tensor, weights: torch.Tensor):
            # Input encoding via Rx rotations
            for i in range(n_qubits):
                qml.RX(inputs[i], wires=i)
            # Variational layers
            for layer in range(n_layers):
                for q in range(n_qubits):
                    qml.RY(weights[layer, q, 0], wires=q)
                    qml.RZ(weights[layer, q, 1], wires=q)
                    qml.RX(weights[layer, q, 2], wires=q)
                # Entanglement pattern (ring)
                for q in range(n_qubits - 1):
                    qml.CNOT(wires=[q, q + 1])
                qml.CNOT(wires=[n_qubits - 1, 0])
            return qml.probs(wires=range(n_qubits))

        self.circuit = circuit

    def __call__(self, inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """
        Evaluate the sampler network.
        :param inputs: array of shape (n_qubits,)
        :param weights: array of shape (n_layers, n_qubits, 3)
        :return: probability distribution over 2^n_qubits states
        """
        inputs_t = torch.tensor(inputs, dtype=torch.float32)
        weights_t = torch.tensor(weights, dtype=torch.float32)
        probs = self.circuit(inputs_t, weights_t)
        return probs.detach().numpy()

__all__ = ["HybridSamplerQNN"]
