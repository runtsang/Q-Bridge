python
import pennylane as qml
import pennylane.numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class QuantumAutoencoderConfig:
    """Configuration for the pure‑quantum autoencoder."""
    n_qubits: int = 8          # total number of qubits
    latent_dim: int = 4        # number of qubits used for encoding
    reps: int = 3              # repetitions of the variational layer
    init: str = "normal"       # initialization scheme for parameters

class Autoencoder:
    """Pure quantum autoencoder implemented with PennyLane."""
    def __init__(self, config: QuantumAutoencoderConfig):
        self.config = config
        self.dev = qml.device("default.qubit", wires=config.n_qubits)

        # QNode that receives the latent vector followed by variational parameters
        self.qnode = qml.QNode(self._circuit, self.dev, interface="autograd")

    def _circuit(self, *params):
        """
        Quantum circuit.
        params: first `latent_dim` entries are the input data,
                the remaining entries are trainable variational parameters.
        """
        latent = params[:self.config.latent_dim]
        weights = params[self.config.latent_dim:]

        # Encode input into the first `latent_dim` qubits
        for i, v in enumerate(latent):
            qml.RX(v, wires=i)

        # Variational block
        qml.templates.StronglyEntanglingLayers(
            weights,
            wires=range(self.config.n_qubits),
            reps=self.config.reps,
        )

        # Return expectation values of PauliZ on all qubits
        return [qml.expval(qml.PauliZ(i)) for i in range(self.config.n_qubits)]

    def __call__(self, latent: np.ndarray) -> np.ndarray:
        """
        Forward pass: latent (batch, latent_dim) → reconstructed vector.
        """
        return self.qnode(*latent.T)

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 200,
        lr: float = 0.01
    ):
        """
        Adam optimizer that updates only the variational parameters.
        The input data is treated as a fixed part of the circuit.
        """
        opt = qml.optimizers.Adam(stepsize=lr)

        # Initialize variational parameters
        param_shape = (self.config.reps * self.config.n_qubits * 3,)
        weights = np.random.randn(*param_shape)

        for _ in range(epochs):
            # Compute loss over the dataset
            loss = 0.0
            for x in data:
                params = np.concatenate([x, weights])
                recon = self.qnode(*params)
                loss += np.mean((recon - x) ** 2)
            loss /= len(data)

            # Gradient w.r.t. variational parameters
            grads = opt.grad(lambda w: self._loss(w, data), weights)
            weights = opt.apply_gradients(zip(grads, [weights]))[0]

        # Store optimized parameters in the QNode
        self.qnode.parameters = weights

    def _loss(self, weights, data):
        loss = 0.0
        for x in data:
            params = np.concatenate([x, weights])
            recon = self.qnode(*params)
            loss += np.mean((recon - x) ** 2)
        return loss / len(data)

__all__ = ["Autoencoder", "QuantumAutoencoderConfig"]
