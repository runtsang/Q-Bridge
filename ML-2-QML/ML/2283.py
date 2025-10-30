"""Python module implementing a classical autoencoder with optional quantum refiner.

The module exposes a single `UnifiedAutoEncoder` class that can be instantiated
with a classical dense auto‑encoder and an optional quantum refiner.  The
class shares the same public API as the quantum implementation so that
downstream code can import either module without modification.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class AutoencoderConfig:
    """Configuration for a classic dense auto‑encoder."""
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64), dropout: float = 0.1):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

class AutoencoderNet(nn.Module):
    """A simple fully‑connected auto‑encoder."""
    def __init__(self, config: AutoencoderConfig):
        super().__init__()
        encoder_layers = []
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, config.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, config.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

class UnifiedAutoEncoder:
    """Classical auto‑encoder with optional quantum refiner.

    The class exposes the same API as the quantum implementation so that
    downstream code can import either module and instantiate a
    `UnifiedAutoEncoder` without modification.  The quantum refiner
    is a callable that accepts a latent tensor and returns a refined
    tensor of identical shape.  The refiner is attached via
    :meth:`attach_quantum_refiner`.  Training is performed only on the
    classical sub‑network; the refiner is assumed to be pre‑trained or
    externally optimised.
    """
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 32,
                 hidden_dims: tuple[int, int] = (128, 64),
                 dropout: float = 0.1,
                 device: torch.device | None = None) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoencoderNet(
            AutoencoderConfig(input_dim, latent_dim, hidden_dims, dropout)
        ).to(self.device)
        self.quantum_refiner = None

    def attach_quantum_refiner(self, refiner: callable) -> None:
        """Attach a callable that refines the latent representation."""
        self.quantum_refiner = refiner

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model.encode(x.to(self.device))
        if self.quantum_refiner is not None:
            z = self.quantum_refiner(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z.to(self.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def train(self,
              data: torch.Tensor,
              *,
              epochs: int = 100,
              batch_size: int = 64,
              lr: float = 1e-3,
              weight_decay: float = 0.0) -> list[float]:
        """Train the classical auto‑encoder.

        Parameters
        ----------
        data : torch.Tensor
            The training data of shape (N, input_dim).
        epochs : int
            Number of training epochs.
        batch_size : int
            Mini‑batch size.
        lr : float
            Optimiser learning rate.
        weight_decay : float
            L2 regularisation strength.
        """
        self.model.train()
        dataset = TensorDataset(data.to(self.device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr,
                                     weight_decay=weight_decay)
        loss_fn = nn.MSELoss()
        history: list[float] = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimizer.zero_grad()
                recon = self.model(batch)
                loss = loss_fn(recon, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history

__all__ = ["UnifiedAutoEncoder", "AutoencoderConfig", "AutoencoderNet"]
