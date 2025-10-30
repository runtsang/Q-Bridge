"""Quantum autoencoder implemented with PennyLane.

The class :class:`AutoencoderNet` encapsulates a variational quantum circuit
that encodes classical data into a latent subspace, decodes it back, and
is trained to minimise a reconstruction loss.  The circuit can be executed
on a simulator or a real quantum backend.
"""

import pennylane as qml
import pennylane.numpy as np
import torch
from torch import nn
from dataclasses import dataclass
from typing import Sequence, Callable, Optional
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class AutoencoderConfig:
    """Configuration for the quantum autoencoder."""
    num_qubits: int
    latent_dim: int = 3
    hidden_layers: Sequence[int] = (4,)
    shots: int = 1000
    seed: int = 42
    device: str = "default.qubit"
    optimizer_lr: float = 0.01
    epochs: int = 200
    batch_size: int = 32
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss()


class AutoencoderNet(nn.Module):
    """Variational quantum autoencoder built with PennyLane."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.dev = qml.device(
            config.device,
            wires=config.num_qubits,
            shots=config.shots,
            seed=config.seed,
        )
        # Initialize trainable weights for the variational layer
        self.weights = nn.Parameter(
            torch.randn(
                len(config.hidden_layers),
                config.num_qubits,
                3,
                dtype=torch.float32,
                requires_grad=True,
            )
        )

        @qml.qnode(self.dev, interface="torch", diff_method="param_shift")
        def circuit(x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
            """Quantum circuit that maps *x* to a latent representation."""
            # Feature embedding
            qml.templates.AngleEmbedding(x, wires=range(config.num_qubits))
            # Variational block
            qml.layers.StronglyEntanglingLayers(weights, wires=range(config.num_qubits))
            # Return expectation values of PauliZ on each qubit
            return torch.stack([qml.expval(qml.PauliZ(i)) for i in range(config.num_qubits)])

        self.circuit = circuit

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent vector for input *x*."""
        return self.circuit(x, self.weights)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode the latent vector back to the input space."""
        # In a VAE the decoder is another quantum circuit; for simplicity we
        # use a linear layer that maps the latent vector to the original dimension.
        # This keeps the implementation lightweight while still demonstrating
        # a hybrid architecture.
        if not hasattr(self, "_decoder"):
            self._decoder = nn.Linear(self.config.num_qubits, self.config.num_qubits)
            nn.init.xavier_uniform_(self._decoder.weight)
        return self._decoder(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        latent = self.encode(x)
        return self.decode(latent)

    def train_autoencoder(
        self,
        data: torch.Tensor,
        *,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        optimizer_lr: Optional[float] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        plot: bool = False,
        early_stop: Optional[int] = None,
    ) -> list[float]:
        """Train the quantum autoencoder and return the loss history."""
        epochs = epochs or self.config.epochs
        batch_size = batch_size or self.config.batch_size
        optimizer_lr = optimizer_lr or self.config.optimizer_lr
        loss_fn = loss_fn or self.config.loss_fn

        optimizer = torch.optim.Adam([self.weights], lr=optimizer_lr)
        dataset = TensorDataset(_as_tensor(data))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        history: list[float] = []
        best_loss = float("inf")
        patience = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                batch = batch.to(self.dev.device)
                optimizer.zero_grad(set_to_none=True)
                recon = self(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)

            if plot:
                plt.ion()
                plt.clf()
                plt.plot(history, label="train loss")
                plt.xlabel("epoch")
                plt.ylabel("MSE")
                plt.legend()
                plt.pause(0.01)

            if early_stop is not None:
                if epoch_loss < best_loss - 1e-6:
                    best_loss = epoch_loss
                    patience = 0
                else:
                    patience += 1
                if patience >= early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if plot:
            plt.ioff()
            plt.show()

        return history


def Autoencoder(
    num_qubits: int,
    *,
    latent_dim: int = 3,
    hidden_layers: Sequence[int] = (4,),
    shots: int = 1000,
    seed: int = 42,
    device: str = "default.qubit",
    optimizer_lr: float = 0.01,
    epochs: int = 200,
    batch_size: int = 32,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
) -> AutoencoderNet:
    """Factory that returns a configured quantum autoencoder."""
    config = AutoencoderConfig(
        num_qubits=num_qubits,
        latent_dim=latent_dim,
        hidden_layers=hidden_layers,
        shots=shots,
        seed=seed,
        device=device,
        optimizer_lr=optimizer_lr,
        epochs=epochs,
        batch_size=batch_size,
        loss_fn=loss_fn,
    )
    return AutoencoderNet(config)


__all__ = [
    "Autoencoder",
    "AutoencoderConfig",
    "AutoencoderNet",
]
