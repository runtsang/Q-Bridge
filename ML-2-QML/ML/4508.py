"""Hybrid autoencoder combining quanvolution, transformer, and regression head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AutoencoderGen192Config:
    """Configuration for the hybrid autoencoder."""

    input_channels: int = 1
    latent_dim: int = 32
    conv_out_channels: int = 4
    conv_kernel: Tuple[int, int] = (2, 2)
    conv_stride: Tuple[int, int] = (2, 2)
    embed_dim: int = 64
    num_transformer_layers: int = 2
    num_heads: int = 4
    ffn_dim: int = 128
    dropout: float = 0.1
    regressor_output: int = 1


class QuanvolutionFilter(nn.Module):
    """Convolutional front‑end that mimics a 2×2 patch extractor."""

    def __init__(self, config: AutoencoderGen192Config) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.conv_out_channels,
            kernel_size=config.conv_kernel,
            stride=config.conv_stride,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.conv(x)


class AutoencoderGen192(nn.Module):
    """Hybrid autoencoder that uses quanvolution, transformer encoder/decoder and a regression head."""

    def __init__(self, config: AutoencoderGen192Config) -> None:
        super().__init__()
        self.config = config
        self.qfilter = QuanvolutionFilter(config)

        # Flattened feature dimension after convolution
        self.flat_dim = config.conv_out_channels * 14 * 14

        # Embedding from flattened features to transformer dimension
        self.embed = nn.Linear(self.flat_dim, config.embed_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_transformer_layers)

        # Latent mapping
        self.latent_linear = nn.Linear(config.embed_dim, config.latent_dim)

        # Decoder: linear to reconstruct flattened features
        self.decoder_linear = nn.Linear(config.latent_dim, self.flat_dim)

        # Reshape to feature map and upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=config.conv_out_channels,
            out_channels=config.input_channels,
            kernel_size=2,
            stride=2,
        )

        # Auxiliary regression head
        self.regressor = nn.Linear(config.latent_dim, config.regressor_output)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, C, H, W)
        batch = x.shape[0]
        features = self.qfilter(x)  # (B, conv_out, H/2, W/2)
        flat = features.view(batch, -1)  # (B, flat_dim)

        # Embed and reshape for transformer
        embedded = self.embed(flat).unsqueeze(1)  # (B, 1, embed_dim)
        encoded = self.transformer(embedded)  # (B, 1, embed_dim)
        encoded = encoded.squeeze(1)  # (B, embed_dim)

        # Latent representation
        latent = self.latent_linear(encoded)  # (B, latent_dim)

        # Reconstruction
        recon_flat = self.decoder_linear(latent)  # (B, flat_dim)
        recon_features = recon_flat.view(batch, self.config.conv_out_channels, 14, 14)
        recon_up = self.upsample(recon_features)  # (B, conv_out, 28, 28)
        reconstruction = self.conv_transpose(recon_up)  # (B, C, 28, 28)

        # Regression output
        regression = self.regressor(latent)  # (B, 1)

        return reconstruction, regression

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation."""
        batch = x.shape[0]
        features = self.qfilter(x)
        flat = features.view(batch, -1)
        embedded = self.embed(flat).unsqueeze(1)
        encoded = self.transformer(embedded).squeeze(1)
        latent = self.latent_linear(encoded)
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent vector."""
        recon_flat = self.decoder_linear(latent)
        recon_features = recon_flat.view(latent.shape[0], self.config.conv_out_channels, 14, 14)
        recon_up = self.upsample(recon_features)
        reconstruction = self.conv_transpose(recon_up)
        return reconstruction


def AutoencoderGen192Factory(
    input_channels: int = 1,
    latent_dim: int = 32,
    conv_out_channels: int = 4,
    conv_kernel: Tuple[int, int] = (2, 2),
    conv_stride: Tuple[int, int] = (2, 2),
    embed_dim: int = 64,
    num_transformer_layers: int = 2,
    num_heads: int = 4,
    ffn_dim: int = 128,
    dropout: float = 0.1,
    regressor_output: int = 1,
) -> AutoencoderGen192:
    """Convenience factory mirroring the original Autoencoder signature."""
    cfg = AutoencoderGen192Config(
        input_channels=input_channels,
        latent_dim=latent_dim,
        conv_out_channels=conv_out_channels,
        conv_kernel=conv_kernel,
        conv_stride=conv_stride,
        embed_dim=embed_dim,
        num_transformer_layers=num_transformer_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
        regressor_output=regressor_output,
    )
    return AutoencoderGen192(cfg)


def train_autoencoder_gen192(
    model: AutoencoderGen192,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop that optimises reconstruction and regression simultaneously."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = torch.utils.data.TensorDataset(_as_tensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    recon_criterion = nn.MSELoss()
    reg_criterion = nn.MSELoss()

    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, reg = model(batch)
            loss_recon = recon_criterion(recon, batch)
            # Dummy target for regression: use mean pixel value
            target_reg = batch.mean(dim=(1, 2, 3), keepdim=True)
            loss_reg = reg_criterion(reg, target_reg)
            loss = loss_recon + loss_reg
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history


def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Utility to ensure input is a float32 tensor on the default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


__all__ = [
    "AutoencoderGen192",
    "AutoencoderGen192Config",
    "AutoencoderGen192Factory",
    "train_autoencoder_gen192",
]
