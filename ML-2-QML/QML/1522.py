"""Autoencoder__gen081.py
Hybrid quantum‑classical variational autoencoder built with PennyLane.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane import qnodes
from pennylane.devices import DefaultQubit
from pennylane.optimize import AdamOptimizer
from typing import Callable, Sequence

# --------------------------------------------------------------------------- #
# Helper: Feature map
# --------------------------------------------------------------------------- #
def _feature_map(data: np.ndarray, num_qubits: int) -> Callable:
    """Return a feature‑map circuit embedding classical data."""
    def circuit(x: np.ndarray):
        for i in range(num_qubits):
            qml.RX(x[i], wires=i)
            qml.RZ(x[i], wires=i)
    return circuit

# --------------------------------------------------------------------------- #
# Helper: Variational ansatz
# --------------------------------------------------------------------------- #
def _var_ansatz(num_qubits: int, reps: int = 2) -> Callable:
    """Return a parameterized ansatz for the latent sub‑space."""
    def circuit(params: np.ndarray):
        idx = 0
        for _ in range(reps):
            for i in range(num_qubits):
                qml.RX(params[idx], wires=i)
                idx += 1
            for i in range(num_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(num_qubits):
                qml.RZ(params[idx], wires=i)
                idx += 1
    return circuit

# --------------------------------------------------------------------------- #
# Hybrid Autoencoder QNode
# --------------------------------------------------------------------------- #
class QuantumAutoEncoder:
    """Variational autoencoder with classical latent extraction."""
    def __init__(
        self,
        num_qubits: int,
        latent_dim: int,
        device: qml.Device | None = None,
        feature_map: Callable | None = None,
        ansatz: Callable | None = None,
        reps: int = 2,
    ):
        self.num_qubits = num_qubits
        self.latent_dim = latent_dim
        self.device = device or DefaultQubit(device='default.qubit', wires=num_qubits)
        self.feature_map = feature_map or _feature_map
        self.ansatz = ansatz or _var_ansatz
        self.reps = reps

        # Parameter vector length
        self.n_params = (reps * (2 * num_qubits))

        # Initialize parameters
        self.params = pnp.random.randn(self.n_params)
        self._build_qnode()

    def _build_qnode(self):
        @qml.qnode(self.device, interface="autograd", diff_method="backprop")
        def circuit(x: np.ndarray, params: np.ndarray):
            # Encode input
            self.feature_map(x, self.num_qubits)(x)
            # Variational ansatz
            self.ansatz(self.num_qubits, self.reps)(params)
            # Measurement: expectation of Z on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

        self.circuit = circuit

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Return latent representation of a single data point."""
        return self.circuit(x, self.params)

    def loss(self, batch: np.ndarray) -> np.ndarray:
        """Mean‑squared reconstruction error over a batch."""
        recon = np.array([self.encode(x) for x in batch])
        return np.mean(np.sum((recon - batch) ** 2, axis=1))

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 200,
        lr: float = 0.01,
        batch_size: int = 32,
        optimizer_cls: Callable = AdamOptimizer,
        callback: Callable | None = None,
        verbose: bool = True,
    ) -> list[float]:
        """Gradient‑based training loop."""
        opt = optimizer_cls(lr)
        history: list[float] = []

        for epoch in range(epochs):
            perm = np.random.permutation(len(data))
            epoch_loss = 0.0
            for i in range(0, len(data), batch_size):
                batch = data[perm[i:i + batch_size]]
                loss_val, grads = qml.gradients.gradients(
                    self.circuit, 0, self.params, batch
                )
                loss_val = loss_val.mean()
                self.params -= opt.step(grads, self.params)
                epoch_loss += loss_val * len(batch)

            epoch_loss /= len(data)
            history.append(epoch_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"[QAE] Epoch {epoch+1} loss: {epoch_loss:.6f}")

            if callback:
                callback(epoch, epoch_loss)

        return history

    def evaluate(self, data: np.ndarray, batch_size: int = 64) -> float:
        """Return average reconstruction MSE."""
        total = 0.0
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            total += np.mean(np.sum((self.encode(batch) - batch) ** 2, axis=1)) * len(batch)
        return total / len(data)

# --------------------------------------------------------------------------- #
# Factory helper
# --------------------------------------------------------------------------- #
def QuantumAutoencoder(
    num_qubits: int,
    latent_dim: int,
    *,
    device: qml.Device | None = None,
    reps: int = 2,
) -> QuantumAutoEncoder:
    """Convenience constructor mirroring the classical Autoencoder factory."""
    return QuantumAutoEncoder(num_qubits, latent_dim, device=device, reps=reps)


__all__ = ["QuantumAutoEncoder", "QuantumAutoencoder"]
