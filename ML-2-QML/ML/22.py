"""Autoencoder with residual connections and latent interpolation for classical data.

This module extends the original fully‑connected autoencoder by adding:
- Residual skip connections between encoder and decoder layers.
- A method for linear interpolation in latent space.
- Early‑stopping training based on a validation set.
- Utility to compute reconstruction error statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, List

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #

def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class AutoencoderConfig:
    """Configuration values for :class:`AutoencoderNet`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    residual: bool = True
    interpolate_steps: int = 10


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class AutoencoderNet(nn.Module):
    """A lightweight multilayer perceptron autoencoder with residual connections."""

    def __init__(self, config: AutoencoderConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder_layers: nn.ModuleList = nn.ModuleList()
        self.decoder_layers: nn.ModuleList = nn.ModuleList()

        # Encoder
        in_dim = config.input_dim
        for hidden in config.hidden_dims:
            self.encoder_layers.append(nn.Linear(in_dim, hidden))
            self.encoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                self.encoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        self.encoder_layers.append(nn.Linear(in_dim, config.latent_dim))

        # Decoder
        in_dim = config.latent_dim
        for hidden in reversed(config.hidden_dims):
            self.decoder_layers.append(nn.Linear(in_dim, hidden))
            self.decoder_layers.append(nn.ReLU())
            if config.dropout > 0.0:
                self.decoder_layers.append(nn.Dropout(config.dropout))
            in_dim = hidden
        self.decoder_layers.append(nn.Linear(in_dim, config.input_dim))

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder."""
        return self.encoder_forward(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Forward pass through the decoder."""
        return self.decoder_forward(latents)

    # --------------------------------------------------------------------- #
    # Internal helpers for residual connections
    # --------------------------------------------------------------------- #

    def encoder_forward(self, x: torch.Tensor) -> torch.Tensor:
        activations = []
        out = x
        for layer in self.encoder_layers:
            out = layer(out)
            # Store activation after each Linear layer (before activation)
            if isinstance(layer, nn.Linear):
                activations.append(out)
        return out, activations

    def decoder_forward(self, z: torch.Tensor) -> torch.Tensor:
        out = z
        idx = 0  # index for skip connections
        for layer in self.decoder_layers:
            out = layer(out)
            # Add skip connection after each Linear layer if configured
            if self.config.residual and isinstance(layer, nn.Linear) and idx < len(self.encoder_layers):
                skip = self.encoder_layers[idx](out)  # linear mapping for skip
                out = out + skip
                idx += 1
        return out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        latent, _ = self.encoder_forward(inputs)
        return self.decoder_forward(latent)

    # --------------------------------------------------------------------- #
    # Latent interpolation
    # --------------------------------------------------------------------- #

    def interpolate(self, z1: torch.Tensor, z2: torch.Tensor) -> List[torch.Tensor]:
        """Return a list of decoded samples along a linear path in latent space."""
        if z1.shape!= z2.shape:
            raise ValueError("Latent vectors must have the same shape.")
        samples = []
        for t in torch.linspace(0.0, 1.0, steps=self.config.interpolate_steps):
            z = (1 - t) * z1 + t * z2
            samples.append(self.decode(z))
        return samples

    # --------------------------------------------------------------------- #
    # Reconstruction error statistics
    # --------------------------------------------------------------------- #

    def compute_reconstruction_error(self, data: torch.Tensor) -> Tuple[float, float]:
        """Return mean and std of MSE over the dataset."""
        with torch.no_grad():
            recon = self.forward(data)
            mse = ((recon - data) ** 2).mean(dim=1)
            return mse.mean().item(), mse.std().item()


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def Autoencoder(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
    residual: bool = True,
    interpolate_steps: int = 10,
) -> AutoencoderNet:
    """Factory that mirrors the quantum helper returning a configured network."""
    config = AutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        residual=residual,
        interpolate_steps=interpolate_steps,
    )
    return AutoencoderNet(config)


# --------------------------------------------------------------------------- #
# Training with early stopping
# --------------------------------------------------------------------------- #

def train_autoencoder(
    model: AutoencoderNet,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    val_fraction: float = 0.1,
    patience: int = 10,
) -> List[float]:
    """
    Simple reconstruction training loop returning the loss history.

    The training now includes:
    - Early stopping based on validation loss.
    - Optional weight decay.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Split into train / val
    dataset = TensorDataset(_as_tensor(data))
    train_size = int(len(dataset) * (1 - val_fraction))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: List[float] = []

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                reconstruction = model(batch)
                loss = loss_fn(reconstruction, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_loader.dataset)

        history.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return history


__all__ = ["Autoencoder", "AutoencoderConfig", "AutoencoderNet", "train_autoencoder"]
