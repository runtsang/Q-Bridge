import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
from typing import Sequence

class AutoencoderGen308:
    """
    Variational autoencoder implemented with PennyLane.
    The encoder is a parameterized quantum circuit that maps
    a classical input vector into a latent representation.
    The decoder is a linear map that reconstructs the input.
    """

    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        reps: int = 3,
        num_layers: int = 2,
        dev: str | qml.Device | None = None,
    ) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.reps = reps
        self.num_layers = num_layers
        self.dev = dev or qml.device("default.qubit", wires=num_qubits)

        # Trainable parameters for the variational ansatz
        self.weights = pnp.random.randn(self.num_layers, self.num_qubits, 3)

        # Decoder parameters
        self.decoder_weights = pnp.random.randn(self.num_qubits, self.latent_dim)
        self.decoder_bias = pnp.random.randn(self.latent_dim)

        # Quantum node
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: Sequence[float]) -> Sequence[float]:
            qml.templates.AngleEmbedding(inputs, wires=range(self.num_qubits))
            qml.templates.StronglyEntanglingLayers(self.weights, wires=range(self.num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def encode(self, inputs: Sequence[float]) -> Sequence[float]:
        """Encode classical data into a latent vector."""
        return self.circuit(inputs)

    def decode(self, latent: Sequence[float]) -> Sequence[float]:
        """Decode latent vector back to classical space."""
        return pnp.dot(latent, self.decoder_weights.T) + self.decoder_bias

    def forward(self, inputs: Sequence[float]) -> Sequence[float]:
        latent = self.encode(inputs)
        return self.decode(latent)

def train_autoencoder_qml(
    model: AutoencoderGen308,
    data: Sequence[Sequence[float]],
    *,
    epochs: int = 200,
    lr: float = 0.01,
    batch_size: int = 16,
    seed: int | None = None,
) -> list[float]:
    """Train the quantum autoencoder with gradient descent."""
    if seed is not None:
        np.random.seed(seed)
    opt = AdamOptimizer(lr)
    loss_history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, len(data), batch_size):
            batch = data[i : i + batch_size]
            batch_inputs = np.array(batch)

            def loss_fn():
                recon = np.array([model.forward(x) for x in batch_inputs])
                return np.mean((recon - batch_inputs) ** 2)

            grads = qml.grad(loss_fn)(model.weights)
            opt.step(grads, model.weights)
            epoch_loss += loss_fn()
        epoch_loss /= np.ceil(len(data) / batch_size)
        loss_history.append(epoch_loss)
    return loss_history

__all__ = ["AutoencoderGen308", "train_autoencoder_qml"]
