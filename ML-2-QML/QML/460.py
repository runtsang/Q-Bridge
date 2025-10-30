import pennylane as qml
import pennylane.numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Callable

@dataclass
class AutoencoderConfig:
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1

class AutoencoderExtended:
    """Quantum autoencoder built with Pennylane."""
    def __init__(self, config: AutoencoderConfig, dev: qml.Device | None = None):
        self.config = config
        self.input_dim = config.input_dim
        self.latent_dim = config.latent_dim
        self.dev = dev or qml.device("default.qubit", wires=self.input_dim)
        # initialise parameters for encoding and decoding
        self.weights_enc = np.random.randn(self.input_dim, 5)
        self.weights_dec = np.random.randn(self.input_dim, 5)

    def _encode_circuit(self, inputs: np.ndarray):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(inputs):
            qml.templates.AngleEmbedding(inputs, wires=range(self.input_dim))
            qml.templates.RealAmplitudes(self.weights_enc, wires=range(self.input_dim))
            return [qml.expval(qml.PauliZ(w)) for w in range(self.input_dim)]
        return circuit(inputs)

    def _decode_circuit(self, latents: np.ndarray):
        @qml.qnode(self.dev, interface="autograd")
        def circuit(latents):
            qml.templates.AngleEmbedding(latents, wires=range(self.input_dim))
            qml.templates.RealAmplitudes(self.weights_dec, wires=range(self.input_dim))
            return [qml.expval(qml.PauliZ(w)) for w in range(self.input_dim)]
        return circuit(latents)

    def encode(self, inputs: np.ndarray) -> np.ndarray:
        """Return the latent representation of the input."""
        return self._encode_circuit(inputs)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Return the reconstruction from the latent codes."""
        return self._decode_circuit(latents)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Encode and immediately decode."""
        latent = self.encode(inputs)
        return self.decode(latent)

    def loss(self, recon: np.ndarray, target: np.ndarray) -> float:
        """Mean‑squared error between reconstruction and target."""
        return np.mean((recon - target) ** 2)

    def train(
        self,
        data: np.ndarray,
        *,
        epochs: int = 100,
        lr: float = 0.01,
        loss_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> List[float]:
        """Train the quantum autoencoder with Adam."""
        if loss_fn is None:
            loss_fn = self.loss
        opt = qml.AdamOptimizer(stepsize=lr)
        params = [self.weights_enc, self.weights_dec]
        history: List[float] = []

        for _ in range(epochs):
            def cost(params):
                self.weights_enc, self.weights_dec = params
                recon = self.forward(data)
                return loss_fn(recon, data)

            params, loss_val = opt.step_and_cost(cost, params)
            history.append(loss_val)
        return history

    def evaluate(self, data: np.ndarray) -> float:
        """Return the average MSE over the data."""
        recon = self.forward(data)
        return np.mean((recon - data) ** 2)

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    dev: qml.Device | None = None,
) -> AutoencoderExtended:
    """Factory that returns a quantum autoencoder instance."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return AutoencoderExtended(config, dev=dev)

__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderExtended",
]
