"""Quantum autoencoder using Pennylane with fidelity loss."""

import pennylane as qml
import numpy as np
from typing import Tuple, List, Optional

class AutoencoderHybrid:
    """Variational quantum autoencoder with encoder/decoder ansatz and fidelity‑based loss."""
    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        hidden_layers: Tuple[int,...] = (4, 4),
        reps: int = 3,
        device: str = "default.qubit",
    ) -> None:
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.reps = reps
        self.dev = qml.device(device, wires=num_qubits)

        # Parameters for encoder and decoder
        self.encoder_params = np.random.randn(num_qubits, reps, 3)
        self.decoder_params = np.random.randn(latent_dim, reps, 3)
        self.optimizer = qml.AdamOptimizer(stepsize=0.1)

    def _feature_map(self, x: np.ndarray) -> None:
        """Encode classical data using angle‑embedding."""
        for i, val in enumerate(x):
            qml.Hadamard(wires=i)
            qml.CRX(val, wires=i)

    def _ansatz(self, params: np.ndarray, wires: np.ndarray) -> None:
        """RealAmplitudes ansatz."""
        qml.templates.RealAmplitudes(params, wires=wires, reps=self.reps)

    def _state_vector(self, x: np.ndarray, params: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Return state vector for a given point and parameter set."""
        @qml.qnode(self.dev)
        def circuit():
            self._feature_map(x)
            self._ansatz(params[0], wires=range(self.num_qubits))
            self._ansatz(params[1], wires=range(self.latent_dim))
            return qml.state()
        return circuit()

    def loss(self, x: np.ndarray) -> float:
        """Fidelity‑based loss for a single data point."""
        # Original state
        @qml.qnode(self.dev)
        def original():
            self._feature_map(x)
            return qml.state()

        orig_state = original()
        # Encoded + decoded state
        encoded_state = self._state_vector(x, (self.encoder_params, self.decoder_params))
        fidelity = np.abs(np.vdot(orig_state, encoded_state)) ** 2
        return 1.0 - fidelity

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 200,
        batch_size: int = 32,
        early_stopping: int = 20,
        verbose: bool = False,
    ) -> List[float]:
        """Gradient‑based training loop returning loss history."""
        history: List[float] = []
        best_loss = float("inf")
        patience = early_stopping

        for epoch in range(epochs):
            epoch_loss = 0.0
            # Shuffle data
            indices = np.random.permutation(len(data))
            for start in range(0, len(data), batch_size):
                batch_idx = indices[start : start + batch_size]
                batch = data[batch_idx]

                def batch_loss(params):
                    loss = 0.0
                    for x in batch:
                        @qml.qnode(self.dev)
                        def circuit():
                            self._feature_map(x)
                            self._ansatz(params[0], wires=range(self.num_qubits))
                            self._ansatz(params[1], wires=range(self.latent_dim))
                            return qml.state()
                        state = circuit()
                        @qml.qnode(self.dev)
                        def original():
                            self._feature_map(x)
                            return qml.state()
                        orig = original()
                        fidelity = np.abs(np.vdot(orig, state)) ** 2
                        loss += 1.0 - fidelity
                    return loss / len(batch)

                grads = qml.grad(batch_loss)(
                    (self.encoder_params, self.decoder_params)
                )
                # Update parameters
                self.encoder_params, self.decoder_params = self.optimizer.step(
                    grads, (self.encoder_params, self.decoder_params)
                )
                epoch_loss += batch_loss((self.encoder_params, self.decoder_params)).item()

            epoch_loss /= len(data)
            history.append(epoch_loss)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience = early_stopping
            else:
                patience -= 1
                if patience <= 0:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} – loss: {epoch_loss:.6f}")

        return history

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Return latent state vector for a data point."""
        @qml.qnode(self.dev)
        def circuit():
            self._feature_map(x)
            self._ansatz(self.encoder_params, wires=range(self.num_qubits))
            return qml.state()
        state = circuit()
        # Return amplitudes of latent qubits
        return state[: 2 ** self.latent_dim]

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vector back to classical representation (placeholder)."""
        raise NotImplementedError("Decoder to classical vector not implemented in this prototype.")

__all__ = ["AutoencoderHybrid"]
