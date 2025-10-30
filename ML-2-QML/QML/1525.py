import pennylane as qml
import torch
import numpy as np
from pennylane import numpy as pnp
from dataclasses import dataclass
from typing import Tuple, List
from pennylane.optimize import AdamOptimizer

class QuantumAutoencoderBase:
    """Base for quantum autoencoder providing device setup."""
    def __init__(self, device: str = "default.qubit"):
        self.dev = qml.device(device, wires=1)
    def __repr__(self):
        return f"<{self.__class__.__name__} on {self.dev}>"

@dataclass
class QuantumAutoencoderConfig:
    latent_dim: int = 3
    num_trash: int = 2
    reps: int = 3

class QuantumAutoencoder(QuantumAutoencoderBase):
    """Variational autoencoder that embeds latent vectors into a single qubit."""
    def __init__(self, config: QuantumAutoencoderConfig):
        super().__init__()
        self.config = config
        self.params = pnp.random.randn(self.num_params)  # will be updated

        @qml.qnode(self.dev, interface="torch")
        def circuit(latent, params):
            """Encode a latent vector into a single qubit."""
            # Prepare computational basis state |0>
            # Apply rotations parameterized by latent and variational params
            qml.RY(latent[0], wires=0)
            qml.RZ(latent[1], wires=0)
            qml.RX(latent[2], wires=0)
            for i, p in enumerate(params):
                qml.RY(p, wires=0)
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    @property
    def num_params(self) -> int:
        return self.config.reps * 3

    def encode(self, latent: torch.Tensor) -> torch.Tensor:
        """Return expectation value as a scalar quantum feature."""
        latent_np = latent.detach().cpu().numpy()
        return torch.tensor(self.circuit(latent_np, self.params), dtype=torch.float32)

def train_qautoencoder(
    model: QuantumAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 200,
    lr: float = 0.05,
    batch_size: int = 32,
) -> List[float]:
    """Train the variational circuit using the parameter‑shift rule."""
    optimizer = AdamOptimizer(lr)
    history: List[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for i in range(0, data.size(0), batch_size):
            batch = data[i : i + batch_size]
            loss = 0.0
            for latent in batch:
                # Simple reconstruction target: latent[0] (arbitrary)
                target = latent[0]
                pred = model.encode(latent)
                loss += (pred - target) ** 2
            loss = loss / batch.size(0)
            grads = qml.grad(model.circuit, argnums=1)(batch[0].detach().cpu().numpy(), model.params)
            optimizer.step(params=model.params, gradients=grads)
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= data.size(0)
        history.append(epoch_loss)
    return history

__all__ = [
    "QuantumAutoencoder",
    "QuantumAutoencoderConfig",
    "train_qautoencoder",
]
