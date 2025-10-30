"""Classical self‑attention autoencoder using PyTorch.

The implementation combines the self‑attention logic from the original
SelfAttention seed with the fully‑connected autoencoder from the
Autoencoder seed.  The public API matches the quantum counterpart
(`SelfAttentionAutoencoder`), allowing interchangeable use in
hybrid pipelines.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from typing import Tuple, Iterable


# ----------------------------------------------------------------------
#  Configuration helpers
# ----------------------------------------------------------------------
@dataclass
class AutoencoderConfig:
    """Config for the MLP autoencoder component."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1


# ----------------------------------------------------------------------
#  Autoencoder network
# ----------------------------------------------------------------------
class AutoencoderNet(nn.Module):
    """Fully‑connected encoder/decoder with optional dropout."""
    def __init__(self, cfg: AutoencoderConfig) -> None:
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = cfg.input_dim
        for h in cfg.hidden_dims:
            enc_layers.append(nn.Linear(in_dim, h))
            enc_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                enc_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = cfg.latent_dim
        for h in reversed(cfg.hidden_dims):
            dec_layers.append(nn.Linear(in_dim, h))
            dec_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                dec_layers.append(nn.Dropout(cfg.dropout))
            in_dim = h
        dec_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


# ----------------------------------------------------------------------
#  Self‑attention helper
# ----------------------------------------------------------------------
class ClassicalSelfAttention:
    """Drop‑in replacement for the original SelfAttention helper."""
    def __init__(self, embed_dim: int) -> None:
        self.embed_dim = embed_dim

    def run(self, rotation_params: np.ndarray, entangle_params: np.ndarray,
            inputs: np.ndarray) -> np.ndarray:
        q = torch.as_tensor(inputs @ rotation_params.reshape(self.embed_dim, -1),
                            dtype=torch.float32)
        k = torch.as_tensor(inputs @ entangle_params.reshape(self.embed_dim, -1),
                            dtype=torch.float32)
        v = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(q @ k.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ v).numpy()


# ----------------------------------------------------------------------
#  Combined module
# ----------------------------------------------------------------------
class SelfAttentionAutoencoder(nn.Module):
    """
    Classic self‑attention followed by an MLP autoencoder.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the attention embedding.
    input_dim : int
        Dimensionality of the input vector to the autoencoder.
    latent_dim : int, default 32
        Size of the latent code in the autoencoder.
    hidden_dims : tuple[int, int], default (128, 64)
        Hidden layer sizes for the autoencoder.
    dropout : float, default 0.1
        Dropout probability in the autoencoder.
    """
    def __init__(self,
                 embed_dim: int,
                 input_dim: int,
                 *,
                 latent_dim: int = 32,
                 hidden_dims: Tuple[int, int] = (128, 64),
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.attention = ClassicalSelfAttention(embed_dim)
        self.autoencoder = AutoencoderNet(AutoencoderConfig(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        ))

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(self, inputs: torch.Tensor,
                rotation_params: torch.Tensor,
                entangle_params: torch.Tensor) -> torch.Tensor:
        """
        Apply self‑attention then autoencoder encoding/decoding.

        Parameters
        ----------
        inputs : torch.Tensor
            Input batch (B, input_dim).
        rotation_params : torch.Tensor
            Parameters for the attention rotation matrix
            (embed_dim * embed_dim,).
        entangle_params : torch.Tensor
            Parameters for the attention entanglement
            (embed_dim - 1,).
        """
        # Classical attention
        attn_out = torch.from_numpy(
            self.attention.run(rotation_params.cpu().numpy(),
                               entangle_params.cpu().numpy(),
                               inputs.cpu().numpy())
        ).to(inputs.device)

        # Autoencoder
        return self.autoencoder(attn_out)

    # ------------------------------------------------------------------
    #  Utility for training
    # ------------------------------------------------------------------
    def train_autoencoder(self,
                          data: torch.Tensor,
                          *,
                          epochs: int = 100,
                          batch_size: int = 64,
                          lr: float = 1e-3,
                          weight_decay: float = 0.0,
                          device: torch.device | None = None) -> list[float]:
        """
        Train only the autoencoder part; attention params are frozen.

        Returns
        -------
        history : list[float]
            Training loss per epoch.
        """
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.autoencoder.to(device)
        dataset = TensorDataset(data.to(device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optim = torch.optim.Adam(self.autoencoder.parameters(),
                                 lr=lr, weight_decay=weight_decay)
        criterion = nn.MSELoss()
        history = []

        for _ in range(epochs):
            epoch_loss = 0.0
            for batch, in loader:
                optim.zero_grad()
                recon = self.forward(batch,
                                     rotation_params=torch.zeros(self.attention.embed_dim**2,
                                                                  device=device),
                                     entangle_params=torch.zeros(self.attention.embed_dim-1,
                                                                device=device))
                loss = criterion(recon, batch)
                loss.backward()
                optim.step()
                epoch_loss += loss.item() * batch.size(0)
            epoch_loss /= len(dataset)
            history.append(epoch_loss)
        return history


__all__ = ["SelfAttentionAutoencoder"]
