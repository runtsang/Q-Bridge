"""Quantum autoencoder using Pennylane with fidelity loss and hybrid training."""
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer
from typing import Tuple, Callable

def _fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Compute state‑vector fidelity."""
    return np.abs(np.vdot(state1, state2)) ** 2


class QuantumAutoencoder:
    """
    Variational quantum autoencoder built with Pennylane.
    The encoder maps an input state into a lower‑dimensional subspace,
    while the decoder attempts to reconstruct the original state.
    """
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_layers: int = 3,
        device: str = "default.qubit",
    ) -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dev = qml.device(device, wires=2 * input_dim)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Encode
            for i in range(self.input_dim):
                qml.AmplitudeEmbedding(
                    features=inputs[i],
                    wires=i,
                    normalize=True,
                )
            for layer in range(self.num_layers):
                for wire in range(2 * self.input_dim):
                    qml.RY(params[layer, wire], wires=wire)
                for wire in range(0, 2 * self.input_dim - 1, 2):
                    qml.CNOT(wires=[wire, wire + 1])

            # Measure
            return qml.state()

        self.circuit = circuit
        n_params = self.num_layers * 2 * self.input_dim
        self.params = np.random.randn(n_params, 2 * self.input_dim)

    def fidelity_loss(self, inputs: np.ndarray) -> float:
        """Compute 1 - fidelity between input and reconstructed state."""
        original = self.circuit(inputs, self.params)
        reconstructed = self.circuit(inputs, self.params)
        return 1.0 - _fidelity(original, reconstructed)

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 1000,
        lr: float = 0.01,
        optimizer_cls: Callable = AdamOptimizer,
    ) -> list[float]:
        """Hybrid training loop with gradient descent."""
        opt = optimizer_cls(lr)
        history: list[float] = []

        for epoch in range(epochs):
            loss = 0.0
            for sample in data:
                loss += self.fidelity_loss(sample)
            loss /= len(data)

            grads = qml.grad(self.fidelity_loss)(data[0])
            self.params = opt.step(self.params, grads)
            history.append(loss.item())
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: loss={loss:.6f}")

        return history

    def reconstruct(self, inputs: np.ndarray) -> np.ndarray:
        """Return the reconstructed state for given inputs."""
        return self.circuit(inputs, self.params)


def AutoencoderQNN(
    input_dim: int,
    latent_dim: int,
    num_layers: int = 3,
    device: str = "default.qubit",
) -> QuantumAutoencoder:
    """Factory that returns a configured quantum autoencoder."""
    return QuantumAutoencoder(input_dim, latent_dim, num_layers, device)


__all__ = ["QuantumAutoencoder", "AutoencoderQNN"]
