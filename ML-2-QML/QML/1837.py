"""Pennylane implementation of a quantum autoencoder with swap‑test reconstruction."""

import pennylane as qml
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Callable


@dataclass
class AutoencoderGen018QConfig:
    num_qubits: int          # total qubits = latent + trash + 1
    latent_dim: int
    trash_dim: int
    reps: int = 5
    device: str = "default.qubit"


class AutoencoderGen018:
    """
    A hybrid quantum autoencoder that encodes classical data into a quantum
    latent state using a variational ansatz and reconstructs via a swap‑test.
    """

    def __init__(
        self,
        config: AutoencoderGen018QConfig,
        encode_ansatz: Callable[[np.ndarray, Sequence[int]], None] | None = None,
        decode_ansatz: Callable[[np.ndarray, Sequence[int]], None] | None = None,
    ) -> None:
        self.config = config
        self.dev = qml.device(config.device, wires=config.num_qubits)
        self.encode_ansatz = encode_ansatz or self._real_amplitudes_ansatz
        self.decode_ansatz = decode_ansatz or self._real_amplitudes_ansatz
        # Parameters for the encoding and decoding ansatzes
        self.encode_params = np.random.randn(self.config.reps * config.latent_dim * 3) * 0.01
        self.decode_params = np.random.randn(self.config.reps * config.latent_dim * 3) * 0.01

    @staticmethod
    def _real_amplitudes_ansatz(params: np.ndarray, wires: Sequence[int]) -> None:
        """Basic entangling layer used as a default ansatz."""
        qml.templates.layers.BasicEntanglerLayers(params, wires=wires)

    def encode(self, input_data: np.ndarray) -> np.ndarray:
        """Encode classical data into a quantum state via a variational circuit."""
        @qml.qnode(self.dev, interface="autograd")
        def circuit(x):
            qml.templates.statevectors.InputState(x, wires=range(self.config.latent_dim))
            self.encode_ansatz(self.encode_params, wires=range(self.config.latent_dim))
            return qml.state()
        return circuit(input_data)

    def decode(self, latent_state: np.ndarray) -> np.ndarray:
        """
        Decode the latent state using a swap‑test against the original input.
        Returns the probability of measuring |0> on the auxiliary qubit.
        """
        @qml.qnode(self.dev, interface="autograd")
        def circuit(latent):
            qml.templates.statevectors.InputState(latent, wires=range(self.config.latent_dim))
            aux = self.config.latent_dim + self.config.trash_dim
            qml.Hadamard(wires=aux)
            for i in range(self.config.trash_dim):
                qml.CSwap(wires=[aux, i, self.config.latent_dim + i])
            qml.Hadamard(wires=aux)
            return qml.probs(wires=aux)
        probs = circuit(latent_state)
        return probs[0]  # probability of |0>

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        latent = self.encode(input_data)
        recon = self.decode(latent)
        return recon


def train_quantum_autoencoder(
    model: AutoencoderGen018,
    data: np.ndarray,
    *,
    epochs: int = 100,
    lr: float = 0.01,
    loss_fn: Callable[[np.ndarray, np.ndarray], float] = lambda y_pred, y_true: np.mean((y_pred - y_true) ** 2),
) -> list[float]:
    """Simple training loop using Pennylane's GradientDescentOptimizer."""
    optimizer = qml.GradientDescentOptimizer(lr)
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x in data:
            def loss():
                recon = model.forward(x)
                return loss_fn(recon, x)

            grads = qml.gradients.gradients(loss, model.encode_params, model.decode_params)
            optimizer.step([model.encode_params, model.decode_params], grads)
            epoch_loss += loss()

        epoch_loss /= len(data)
        history.append(epoch_loss)

    return history


__all__ = [
    "AutoencoderGen018",
    "AutoencoderGen018QConfig",
    "train_quantum_autoencoder",
]
