"""Hybrid attention-based autoencoder combining classical encoder, self‑attention, and decoder.

The module extends the original Autoencoder by inserting a trainable self‑attention
block between the encoder and decoder.  This mirrors the quantum self‑attention
pattern while remaining fully classical, enabling direct comparison of
performance and training dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Import primitives from the anchor seed
from Autoencoder import AutoencoderNet, AutoencoderConfig
from SelfAttention import SelfAttention

def _as_tensor(data: torch.Tensor | Iterable[float]) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor

class AttentionLayer(nn.Module):
    """Wraps the classical self‑attention helper as a PyTorch module."""
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Trainable parameters for the attention block
        self.rotation_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.entangle_params = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.attention = SelfAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rotation = self.rotation_params.reshape(self.embed_dim, -1)
        entangle = self.entangle_params.reshape(self.embed_dim, -1)
        # Use the external helper; convert tensors to NumPy for compatibility
        return torch.as_tensor(
            self.attention.run(rotation.cpu().numpy(), entangle.cpu().numpy(), x.cpu().numpy()),
            device=x.device,
            dtype=torch.float32,
        )

@dataclass
class HybridAttentionAutoencoderConfig(AutoencoderConfig):
    """Configuration for the hybrid attention autoencoder."""
    # Inherits all fields from AutoencoderConfig

class HybridAttentionAutoencoder(nn.Module):
    """Hybrid autoencoder with an attention bottleneck."""
    def __init__(self, config: HybridAttentionAutoencoderConfig) -> None:
        super().__init__()
        self.encoder = AutoencoderNet(config)
        self.attention = AttentionLayer(config.latent_dim)
        # Re‑use the decoder from the original architecture
        self.decoder = nn.Sequential(
            *list(AutoencoderNet(config).decoder.children())
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encoder.encode(inputs)
        return self.attention(latent)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))

def HybridAttentionAutoencoderFactory(
    input_dim: int,
    *,
    latent_dim: int = 32,
    hidden_dims: Tuple[int, int] = (128, 64),
    dropout: float = 0.1,
) -> HybridAttentionAutoencoder:
    """Factory that returns a configured hybrid attention autoencoder."""
    config = HybridAttentionAutoencoderConfig(
        input_dim=input_dim,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    return HybridAttentionAutoencoder(config)

def train_hybrid_attention_autoencoder(
    model: HybridAttentionAutoencoder,
    data: torch.Tensor,
    *,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> list[float]:
    """Training loop identical to the original but for the hybrid model."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(batch)
            loss = loss_fn(reconstruction, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
    return history

__all__ = [
    "HybridAttentionAutoencoder",
    "HybridAttentionAutoencoderFactory",
    "HybridAttentionAutoencoderConfig",
    "train_hybrid_attention_autoencoder",
]
