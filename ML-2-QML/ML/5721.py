"""Hybrid classical‑quantum auto‑encoder and classifier.

This module defines the class `QuanvolutionAutoEncoder` that integrates
- a quantum‑enhanced quanvolution filter (patch‑wise 2×2 kernels),
- a classical MLP encoder that projects the concatenated patch features into a latent space,
- a classical decoder that reconstructs the original image, and
- a final linear head that produces class logits.

The design fuses the classical and quantum seeds while remaining fully PyTorch‑compatible.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Iterable

# Import the quantum filter from the QML module
from.qml_module import QFilter


@dataclass
class QuanvolutionAutoEncoderConfig:
    input_channels: int = 1
    patch_size: int = 2
    conv_out_channels: int = 4
    encoder_hidden_dims: Tuple[int,...] = (128, 64)
    decoder_hidden_dims: Tuple[int,...] = (64, 128)
    latent_dim: int = 32
    dropout: float = 0.1
    num_classes: int = 10


class QuanvolutionAutoEncoder(nn.Module):
    def __init__(self, cfg: QuanvolutionAutoEncoderConfig = QuanvolutionAutoEncoderConfig()):
        super().__init__()
        self.cfg = cfg

        # Quantum quanvolution filter
        self.qfilter = QFilter(n_wires=4, random_layers=8)

        # Compute number of patches for MNIST 28x28
        self.num_patches = (28 // cfg.patch_size) ** 2
        self.qfilter_output_dim = cfg.conv_out_channels * self.num_patches

        # Encoder
        encoder_layers = []
        in_dim = self.qfilter_output_dim
        for hidden in cfg.encoder_hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                encoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.decoder_hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, self.qfilter_output_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # Classification head
        self.classifier = nn.Linear(cfg.latent_dim, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        qfeat = self.qfilter(x)
        latent = self.encoder(qfeat)
        logits = self.classifier(latent)
        return F.log_softmax(logits, dim=-1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        qfeat = self.qfilter(x)
        return self.encoder(qfeat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        qfeat_recon = self.decode(z)
        return qfeat_recon


def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


def train_autoencoder(
    model: QuanvolutionAutoEncoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Simple reconstruction training loop returning the loss history."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon = model.reconstruct(batch)
            loss = loss_fn(recon, model.qfilter(batch))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


__all__ = ["QuanvolutionAutoEncoder", "train_autoencoder", "QuanvolutionAutoEncoderConfig"]
