import pennylane as qml
import numpy as np
from autograd import grad

class SamplerQNN:
    """
    Variational quantum sampler network that maps a 2‑dimensional input
    to a probability distribution over 2 measurement outcomes.
    The circuit uses parameterized Ry rotations, a CNOT entanglement
    layer, and supports an autograd interface for end‑to‑end training.
    """

    def __init__(self, n_qubits: int = 2, n_layers: int = 2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Initialize trainable weights
        self.weights = np.random.uniform(0, 2 * np.pi, (n_layers, n_qubits))

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
            # Encode inputs via Ry rotations
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            # Parameterized rotation layers
            for layer in range(n_layers):
                for qubit in range(n_qubits):
                    qml.RY(weights[layer, qubit], wires=qubit)
                # Entanglement
                for qubit in range(n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            # Return probabilities of first two computational basis states
            probs = qml.probs(wires=range(n_qubits))
            return probs[:2]

        self.circuit = circuit

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """
        Compute the output probability distribution for a given 2‑dimensional input.
        """
        return self.circuit(inputs, self.weights)

    def loss(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Cross‑entropy loss between the predicted distribution and a one‑hot target.
        """
        preds = self.forward(inputs)
        eps = 1e-12
        preds = np.clip(preds, eps, 1.0)
        return -np.sum(targets * np.log(preds))

    def gradient(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the trainable weights.
        """
        loss_grad = grad(lambda w: self.loss(inputs, targets))
        return loss_grad(self.weights)
