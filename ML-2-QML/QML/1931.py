"""AutoencoderGen119Q – a hybrid variational autoencoder built with PennyLane."""

import pennylane as qml
from pennylane import numpy as np
import numpy as _np
from typing import Callable, Tuple

# --------------------------------------------------------------------------- #
# Feature mapping – inject classical data into qubit amplitudes
# --------------------------------------------------------------------------- #
def feature_map(data: np.ndarray, n_qubits: int) -> qml.QNode:
    """Return a QNode that encodes data into a state using a simple ZZ feature map."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="parameter_shift")
    def circuit(x):
        # Basic ZZ feature map
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for i in range(n_qubits):
            qml.RX(x[i], wires=i)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        return qml.state()
    return circuit


# --------------------------------------------------------------------------- #
# Parameterized encoder–decoder circuit
# --------------------------------------------------------------------------- #
def variational_circuit(
    latent_dim: int,
    trash_dim: int,
    n_qubits: int,
    weights: np.ndarray,
) -> qml.QNode:
    """Return a QNode implementing encoder + decoder with a swap‑test style readout."""
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="autograd", diff_method="parameter_shift")
    def circuit(data):
        # Encode data (using the feature map)
        for i in range(n_qubits):
            qml.RY(data[i], wires=i)

        # Parameterised layers (encoder)
        offset = 0
        for i in range(latent_dim + trash_dim):
            for j in range(n_qubits):
                qml.Rot(weights[offset, 0], weights[offset, 1], weights[offset, 2], wires=j)
                offset += 1

        # Swap‑test style measurement
        for i in range(trash_dim):
            qml.CSWAP(wires=[0, latent_dim + i, latent_dim + trash_dim + i])

        # Measurement on auxiliary qubit
        return qml.expval(qml.PauliZ(0))
    return circuit


# --------------------------------------------------------------------------- #
# Hybrid Autoencoder wrapper
# --------------------------------------------------------------------------- #
class AutoencoderGen119Q:
    """
    Hybrid quantum autoencoder that:
    * encodes classical data into a quantum circuit via a feature map
    * learns a latent representation in a subset of qubits
    * decodes back to a classical vector through a measurement dictionary
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 3,
        trash_dim: int = 2,
        n_qubits: int | None = None,
        weight_shape: Tuple[int,...] | None = None,
        loss_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.trash_dim = trash_dim
        self.n_qubits = n_qubits or (latent_dim + 2 * trash_dim + 1)
        self.loss_fn = loss_fn or self._mse_loss

        # Initialize trainable parameters
        if weight_shape is None:
            weight_shape = (self.latent_dim + trash_dim, self.n_qubits, 3)
        self.weights = np.random.randn(*weight_shape, requires_grad=True)

        # Build circuits
        self.feature_circuit = feature_map(np.zeros(self.input_dim), self.n_qubits)
        self.vqc = variational_circuit(self.latent_dim, self.trash_dim, self.n_qubits, self.weights)

    # --------------------------------------------------------------------- #
    # Forward pass – returns reconstructed vector
    # --------------------------------------------------------------------- #
    def forward(self, data: np.ndarray) -> np.ndarray:
        """Encode → latent → decode → reconstruction."""
        # 1. Encode classical data
        encoded = self.feature_circuit(data)

        # 2. Pass through variational circuit
        # (the circuit returns a scalar, we map it back to a vector via a linear layer)
        y = self.vqc(encoded)
        # Simple linear decode for demo purposes
        return y.reshape(-1)

    # --------------------------------------------------------------------- #
    # Training utilities
    # --------------------------------------------------------------------- #
    def loss(self, recon: np.ndarray, target: np.ndarray) -> float:
        return self.loss_fn(recon, target)

    @staticmethod
    def _mse_loss(x: np.ndarray, y: np.ndarray) -> float:
        return float(_np.mean((x - y) ** 2))

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 200,
        lr: float = 0.01,
        optimizer: str = "adam",
    ) -> list[float]:
        """Simple gradient‑descent training loop."""
        history: list[float] = []
        if optimizer == "adam":
            opt = qml.AdamOptimizer(stepsize=lr)
        else:
            opt = qml.GradientDescentOptimizer(stepsize=lr)

        for epoch in range(epochs):
            def cost_fn(weights):
                self.weights = weights
                recon = self.forward(data)
                return self.loss(recon, data)

            self.weights = opt.step(cost_fn, self.weights)
            loss_val = cost_fn(self.weights)
            history.append(loss_val)
        return history

    def encode(self, data: np.ndarray) -> np.ndarray:
        """Return latent representation (placeholder, returns zero vector)."""
        # In a real implementation we would measure the latent qubits directly.
        return np.zeros(self.latent_dim)

    def decode(self, latent: np.ndarray) -> np.ndarray:
        """Decode latent vector to reconstruction (placeholder)."""
        return latent  # Identity for illustration


def AutoencoderGen119QFactory(
    input_dim: int,
    *,
    latent_dim: int = 3,
    trash_dim: int = 2,
    n_qubits: int | None = None,
) -> AutoencoderGen119Q:
    """Convenience factory mirroring the original API."""
    return AutoencoderGen119Q(
        input_dim=input_dim,
        latent_dim=latent_dim,
        trash_dim=trash_dim,
        n_qubits=n_qubits,
    )


__all__ = [
    "AutoencoderGen119Q",
    "AutoencoderGen119QFactory",
]
