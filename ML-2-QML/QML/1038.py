"""Quantum autoencoder implementation using Pennylane.

The class builds a variational circuit with a Real‑Amplitudes ansatz for both
encoding and decoding.  Gradient‑based training is provided via a lightweight
wrapper that accepts any torch‑compatible loss function.
"""

import pennylane as qml
import numpy as np
import torch
from torch import nn
from typing import Callable, Iterable, List

class VariationalAutoencoderQML(nn.Module):
    """Variational quantum autoencoder with tunable ansatz depth."""
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        ansatz_depth: int = 3,
        device: str = "default.qubit",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.ansatz_depth = ansatz_depth
        self.dev = qml.device(device, wires=max(input_dim, latent_dim))
        self.input_wires = list(range(input_dim))
        self.latent_wires = list(range(latent_dim))

        # Trainable parameters for encoder and decoder
        self.encoder_params = nn.Parameter(
            torch.randn(ansatz_depth, input_dim, 3)
        )
        self.decoder_params = nn.Parameter(
            torch.randn(ansatz_depth, latent_dim, 3)
        )

    def _ansatz(self, params: torch.Tensor, wires: List[int]) -> None:
        """Apply a layered Real‑Amplitudes ansatz."""
        for layer in range(self.ansatz_depth):
            for w in wires:
                qml.RY(params[layer, w, 0], wires=w)
                qml.RZ(params[layer, w, 1], wires=w)
                qml.RX(params[layer, w, 2], wires=w)
            for (w1, w2) in zip(wires[:-1], wires[1:]):
                qml.CNOT([w1, w2])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode classical data to a latent vector via measurement."""
        @qml.qnode(self.dev, interface="torch")
        def circuit(x):
            qml.templates.AmplitudeEmbedding(
                features=x,
                wires=self.input_wires,
                normalize=True,
            )
            self._ansatz(self.encoder_params, self.input_wires)
            return [qml.expval(qml.PauliZ(w)) for w in self.latent_wires]

        latent = circuit(x)
        return latent

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector back to reconstruction."""
        @qml.qnode(self.dev, interface="torch")
        def circuit(z):
            qml.templates.AmplitudeEmbedding(
                features=z,
                wires=self.latent_wires,
                normalize=True,
            )
            self._ansatz(self.decoder_params, self.latent_wires)
            return [qml.expval(qml.PauliZ(w)) for w in self.input_wires]

        recon = circuit(z)
        return recon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

def train_qautoencoder(
    model: VariationalAutoencoderQML,
    data: Iterable[torch.Tensor],
    *,
    epochs: int = 100,
    lr: float = 0.01,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = lambda y_pred, y_true: torch.mean((y_pred - y_true) ** 2),
    optimizer_cls: Callable = torch.optim.Adam,
) -> List[float]:
    """Train the quantum autoencoder with gradient descent."""
    optimizer = optimizer_cls(
        [model.encoder_params, model.decoder_params], lr=lr
    )
    history: List[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for x in data:
            x_t = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
            z = model.encode(x_t)
            recon = model.decode(z)
            loss = loss_fn(recon, x_t)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(data)
        history.append(epoch_loss)

    return history

__all__ = [
    "VariationalAutoencoderQML",
    "train_qautoencoder",
]
