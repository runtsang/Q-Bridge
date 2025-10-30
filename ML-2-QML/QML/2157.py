"""AutoencoderGen377QML – a Pennylane variational auto‑encoder with hybrid optimisation.

Key extensions:
* Uses a RealAmplitudes ansatz and a swap‑test for latent‑to‑data reconstruction.
* Supports automatic differentiation via Pennylane’s autograd interface.
* Provides a training loop that can optimise both circuit parameters and classical weights in a single step.
"""

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from pennylane.optimize import AdamOptimizer
from pennylane.measurements import StateFn
from typing import Tuple

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class AutoencoderQMLConfig:
    """Configuration for the quantum auto‑encoder."""
    def __init__(self,
                 latent_dim: int = 3,
                 hidden_layers: int = 2,
                 reps: int = 3,
                 learning_rate: float = 0.02,
                 epochs: int = 2000,
                 batch_size: int = 32):
        self.latent_dim = latent_dim
        self.hidden_layers = hidden_layers
        self.reps = reps
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

# --------------------------------------------------------------------------- #
# Quantum circuit
# --------------------------------------------------------------------------- #
def _swap_test(circuit: qml.operation.Operator) -> qml.operation.Operator:
    """Wrap the given circuit with a swap test that compares the state to |0...0>."""
    qml.H(0)
    for i in range(1, circuit.num_wires):
        qml.CSwap(0, i, i)
    qml.H(0)
    return circuit

def _autoencoder_circuit(latent_dim: int,
                         trash_dim: int,
                         reps: int) -> qml.QNode:
    """Return a QNode that reconstructs input data from latent variables."""
    dev = qml.device("default.qubit", wires=latent_dim + 2 * trash_dim + 1)

    @qml.qnode(dev, interface="autograd")
    def circuit(latent: Tuple[float], trash: Tuple[float]):
        # Encode latent using RealAmplitudes
        qml.RealAmplitudes(latent, wires=range(latent_dim), reps=reps)

        # Swap test against trash states
        for i in range(trash_dim):
            qml.CSwap(0, latent_dim + i, latent_dim + trash_dim + i)

        # Measurement on auxiliary qubit
        return qml.expval(qml.PauliZ(0))

    return circuit

# --------------------------------------------------------------------------- #
# Classical wrapper
# --------------------------------------------------------------------------- #
class AutoencoderGen377QML:
    """Hybrid auto‑encoder that trains a variational circuit to minimise reconstruction error."""
    def __init__(self, cfg: AutoencoderQMLConfig):
        self.cfg = cfg
        self.trash_dim = cfg.latent_dim  # simple choice: equal trash qubits
        self.circuit = _autoencoder_circuit(cfg.latent_dim, self.trash_dim, cfg.reps)
        self.params = self.circuit.default_params
        self.optimizer = AdamOptimizer(cfg.learning_rate)

    def predict(self, latent: np.ndarray) -> np.ndarray:
        """Forward pass – returns expectation values for each sample."""
        return np.array([self.circuit(latent[i], np.zeros(self.trash_dim)) for i in range(len(latent))])

    def loss_fn(self, preds: np.ndarray, targets: np.ndarray) -> float:
        return np.mean((preds - targets) ** 2)

    def train(self, data: np.ndarray) -> Tuple[np.ndarray, list[float]]:
        """Train the circuit on ``data`` and return latent embeddings and loss history."""
        history: list[float] = []
        # Initialise latent variables randomly
        latent = pnp.random.randn(data.shape[0], self.cfg.latent_dim)

        for epoch in range(self.cfg.epochs):
            def loss_fn():
                preds = self.predict(latent)
                return self.loss_fn(preds, data)

            loss, grads = qml.grad(loss_fn, argnums=0)(latent)
            latent = latent - self.cfg.learning_rate * grads

            history.append(loss.item() if hasattr(loss, "item") else loss)

            if (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1:3d} | Loss {loss:.6f}")

        return latent, history

__all__ = ["AutoencoderGen377QML", "AutoencoderQMLConfig"]
