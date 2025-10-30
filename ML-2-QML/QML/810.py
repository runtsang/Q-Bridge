"""Quantum autoencoder using Pennylane with a variational feature map and ansatz."""
from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.measurements import StateExpectation
from pennylane.optimize import AdamOptimizer
from typing import Sequence, Callable, Tuple, Optional


def _state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """Return the fidelity between two pure states."""
    return np.abs(np.vdot(state1, state2)) ** 2


class QuantumAutoencoder:
    """
    Variational quantum autoencoder.

    Parameters
    ----------
    num_qubits : int
        Total number of qubits in the circuit (latent + trash).
    latent_dim : int
        Number of qubits used to store the latent representation.
    feature_map : Callable[[np.ndarray], qml.QNode], optional
        Feature map that embeds classical data into a quantum state.
    ansatz : Callable[[int, int], qml.QNode], optional
        Parameterised ansatz that transforms the latent space.
    device : str, optional
        Pennylane device name.
    shots : int, optional
        Number of shots for state sampling.
    """

    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        feature_map: Optional[Callable[[np.ndarray], qml.QNode]] = None,
        ansatz: Optional[Callable[[int, int], qml.QNode]] = None,
        device: str = "default.qubit",
        shots: int = 1024,
    ) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.trash_dim = num_qubits - latent_dim
        self.device = qml.device(device, wires=num_qubits, shots=shots)

        # Default feature map: PauliFeatureMap
        if feature_map is None:
            self.feature_map = qml.templates.PauliFeatureMap
        else:
            self.feature_map = feature_map

        # Default ansatz: StronglyEntanglingLayers
        if ansatz is None:
            self.ansatz = qml.templates.StronglyEntanglingLayers
        else:
            self.ansatz = ansatz

        # Parameter initialization
        self.num_params = self.latent_dim * self.trash_dim
        self.params = np.random.randn(self.num_params)

        # Define the QNode
        @qml.qnode(self.device, interface="autograd")
        def circuit(inputs: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Encode input data
            self.feature_map(inputs, wires=range(self.latent_dim))
            # Apply ansatz
            self.ansatz(params, wires=range(self.num_qubits))
            # Return the full statevector
            return StateExpectation()(qml.state())

        self.circuit = circuit

    def encode(self, inputs: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Return the latent state for *inputs* using current parameters."""
        params = params if params is not None else self.params
        full_state = self.circuit(inputs, params)
        # Extract the first `latent_dim` qubits as the latent state
        return full_state[: self.latent_dim]

    def decode(self, latent_state: np.ndarray, params: np.ndarray | None = None) -> np.ndarray:
        """Reconstruct the input from *latent_state*."""
        params = params if params is not None else self.params
        # Build a circuit that maps latent_state back to full state
        @qml.qnode(self.device, interface="autograd")
        def recon_circ(latent: np.ndarray, params: np.ndarray) -> np.ndarray:
            # Prepare latent qubits
            qml.BasisState(latent, wires=range(self.latent_dim))
            # Apply inverse ansatz
            self.ansatz(params, wires=range(self.num_qubits))
            return StateExpectation()(qml.state())

        return recon_circ(latent_state, params)

    def loss(self, inputs: np.ndarray, params: np.ndarray | None = None) -> float:
        """Reconstruction loss based on fidelity."""
        params = params if params is not None else self.params
        latent = self.encode(inputs, params)
        recon = self.decode(latent, params)
        return 1.0 - _state_fidelity(inputs, recon)

    def train(
        self,
        data: Sequence[np.ndarray],
        *,
        epochs: int = 200,
        lr: float = 0.01,
        optimizer_cls: Callable[..., AdamOptimizer] = AdamOptimizer,
        verbose: bool = True,
    ) -> list[float]:
        """Train the autoencoder on *data*."""
        opt = optimizer_cls(lr)
        history: list[float] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for sample in data:
                loss_val = self.loss(sample, self.params)
                grads = qml.grad(self.loss)(sample, self.params)
                self.params = opt.step(grads, self.params)
                epoch_loss += loss_val
            epoch_loss /= len(data)
            history.append(epoch_loss)
            if verbose and (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} – Loss: {epoch_loss:.6f}")

        return history

    def encode_batch(self, batch: Sequence[np.ndarray]) -> np.ndarray:
        """Encode a batch of samples."""
        return np.array([self.encode(x) for x in batch])

    def reconstruct_batch(self, batch: Sequence[np.ndarray]) -> np.ndarray:
        """Reconstruct a batch of samples."""
        return np.array([self.decode(self.encode(x)) for x in batch])

__all__ = ["QuantumAutoencoder"]
