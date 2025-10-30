"""SamplerQNN – quantum variational sampler implementation.

Key extensions:
* Uses Pennylane for automatic differentiation.
* Supports input‑conditioned rotations and a trainable weight layer.
* Provides a `sample` method that returns a probability distribution
  over the two computational basis states.
"""

import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class SamplerQNN:
    """
    Variational sampler circuit with two qubits.
    
    Parameters
    ----------
    n_qubits : int, default=2
        Number of qubits in the sampler.
    weight_dim : int, default=4
        Number of trainable weight parameters.
    """
    def __init__(self, n_qubits: int = 2, weight_dim: int = 4) -> None:
        self.n_qubits = n_qubits
        self.weight_dim = weight_dim
        self.dev = qml.device("default.qubit", wires=n_qubits)
        # Initialise weights randomly
        self.weights = np.random.uniform(0, 2 * np.pi, weight_dim)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs, weights):
            # Input‑conditioned rotations
            qml.RY(inputs[0], wires=0)
            qml.RY(inputs[1], wires=1)
            # Entangling layer
            qml.CNOT(wires=[0, 1])
            # Trainable rotation layer
            qml.RY(weights[0], wires=0)
            qml.RY(weights[1], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            return qml.probs(wires=[0, 1])

        self.circuit = circuit

    def sample(self, inputs: np.ndarray) -> np.ndarray:
        """
        Return the probability distribution over |00>, |01>, |10>, |11>.
        
        Parameters
        ----------
        inputs : array_like, shape (2,)
            Input parameters to condition the circuit.
        """
        probs = self.circuit(np.array(inputs, dtype=np.float64), self.weights)
        return probs

    def loss(self, inputs: np.ndarray, target: np.ndarray) -> float:
        """
        Cross‑entropy loss between sampled distribution and target.

        Parameters
        ----------
        inputs : array_like, shape (2,)
            Input parameters.
        target : array_like, shape (4,)
            One‑hot target distribution.
        """
        probs = self.sample(inputs)
        # Avoid log(0)
        eps = 1e-12
        return -np.sum(target * np.log(probs + eps))

    def train(
        self,
        data: list[tuple[np.ndarray, np.ndarray]],
        epochs: int = 10,
        lr: float = 0.01,
    ) -> None:
        """
        Train the weight parameters using gradient descent.

        Parameters
        ----------
        data : list of (inputs, target) tuples
            Each target is a one‑hot vector over the four basis states.
        epochs : int
            Number of training epochs.
        lr : float
            Learning rate.
        """
        opt = qml.GradientDescentOptimizer(lr)
        for _ in range(epochs):
            for inputs, target in data:
                self.weights = opt.step(lambda w: self.loss(inputs, target), self.weights)
