"""Quantum autoencoder implemented with Pennylane.

Provides:
    * :class:`Autoencoder` – a variational autoencoder that encodes an input vector
      into a latent representation and reconstructs it.
    * :func:`train_autoencoder_qml` – gradient‑based training routine returning
      the optimized parameters and loss history.
"""

import pennylane as qml
import pennylane.numpy as np  # Autograd‑compatible NumPy
from typing import Iterable, Tuple, List, Optional


class Autoencoder:
    """Variational quantum autoencoder.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input vector.
    latent_dim : int, default 3
        Number of qubits used to store the latent representation.
    device : str, default "default.qubit"
        Pennylane device used for simulation.
    """

    def __init__(self, input_dim: int, latent_dim: int = 3, device: str = "default.qubit") -> None:
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device
        self.wires = list(range(input_dim))
        self.qnode = qml.QNode(self._circuit, qml.device(device, wires=self.wires), interface="autograd")

    def _circuit(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Quantum circuit that maps *x* to a reconstructed vector."""
        # Feature map
        qml.templates.RealAmplitudes(x, wires=self.wires, reps=1)
        # Encoder: first `latent_dim` parameters
        for i in range(self.latent_dim):
            qml.RY(params[i], wires=self.wires[i])
        # Decoder: remaining parameters
        for i in range(self.latent_dim, self.input_dim):
            qml.RY(params[i], wires=self.wires[i])
        # Measurement
        return [qml.expval(qml.PauliZ(i)) for i in self.wires]

    def encode(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Return the latent representation of *x* using *params*."""
        return params[: self.latent_dim]

    def decode(self, latent: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Reconstruct *x* from a latent vector and the full parameter set."""
        new_params = params.copy()
        new_params[: self.latent_dim] = latent
        return self.qnode(np.zeros(self.input_dim), new_params)

    def forward(self, x: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Full forward pass: encode then decode."""
        return self.qnode(x, params)


def train_autoencoder_qml(
    data: Iterable[Iterable[float]],
    input_dim: int,
    latent_dim: int = 3,
    epochs: int = 100,
    lr: float = 0.01,
    device: str = "default.qubit",
    verbose: bool = False,
) -> Tuple[np.ndarray, List[float]]:
    """Train a quantum autoencoder on *data*.

    Returns the optimized parameters and the loss history.
    """
    data_np = np.array(list(data), dtype=np.float64)
    n_samples = data_np.shape[0]
    # Initialise parameters
    params = np.random.randn(input_dim)
    opt = qml.GradientDescentOptimizer(stepsize=lr)
    loss_history: List[float] = []

    ae = Autoencoder(input_dim, latent_dim, device)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x in data_np:
            def loss_fn(p):
                y = ae._circuit(x, p)
                return np.mean((y - x) ** 2)

            loss = loss_fn(params)
            params = opt.step(loss_fn, params)
            epoch_loss += loss
        epoch_loss /= n_samples
        loss_history.append(epoch_loss)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} loss={epoch_loss:.4f}")

    return params, loss_history


__all__ = ["Autoencoder", "train_autoencoder_qml"]
