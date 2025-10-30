"""Hybrid autoencoder that combines classical MLP and transformer encoders
with an optional quantum sampler for stochastic reconstruction.

The class exposes a flexible API that can be configured to use:
- A plain MLP encoder/decoder (default)
- A transformer encoder (if use_transformer_encoder=True)
- A quantum-inspired stochastic sampler that injects noise into the latent
  representation (use_quantum_sampler=True)

The implementation is fully classical and relies only on PyTorch and
NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _as_tensor(data: Iterable[float] | torch.Tensor) -> torch.Tensor:
    """Return a float32 tensor on the current default device."""
    if isinstance(data, torch.Tensor):
        tensor = data
    else:
        tensor = torch.as_tensor(data, dtype=torch.float32)
    if tensor.dtype!= torch.float32:
        tensor = tensor.to(dtype=torch.float32)
    return tensor


@dataclass
class UnifiedAutoEncoderConfig:
    """Configuration for :class:`UnifiedAutoEncoder`."""
    input_dim: int
    latent_dim: int = 32
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.1
    use_transformer_encoder: bool = False
    transformer_num_heads: int = 4
    transformer_ffn_dim: int = 256
    transformer_num_blocks: int = 2
    use_quantum_sampler: bool = False
    sampler_std: float = 0.1


class MultiHeadAttentionClassical(nn.Module):
    """Standard multi‑head attention implemented with PyTorch."""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if embed_dim % num_heads!= 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        return attn_output


class FeedForwardClassical(nn.Module):
    """Two‑layer feed‑forward network."""
    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim)
        self.linear2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


class TransformerBlockClassical(nn.Module):
    """Transformer block composed of attention and feed‑forward."""
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 ffn_dim: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttentionClassical(embed_dim, num_heads, dropout)
        self.ffn = FeedForwardClassical(embed_dim, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, embed_dim: int, max_len: int = 5000) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) *
                             (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class UnifiedAutoEncoder(nn.Module):
    """Hybrid autoencoder with a classical encoder (MLP or transformer) and
    classical decoder.  An optional quantum sampler can be used for
    stochastic latent sampling.
    """
    def __init__(self, cfg: UnifiedAutoEncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg

        # ---------- Encoder ----------
        if cfg.use_transformer_encoder:
            # Build a transformer encoder: token embedding + positional + blocks
            self.token_embed = nn.Linear(cfg.input_dim, cfg.input_dim)
            self.pos_embed = PositionalEncoder(cfg.input_dim)
            self.transformer = nn.Sequential(
                *[TransformerBlockClassical(cfg.input_dim,
                                            cfg.transformer_num_heads,
                                            cfg.transformer_ffn_dim,
                                            cfg.dropout)
                  for _ in range(cfg.transformer_num_blocks)]
            )
            # Project to latent space
            self.encoder_proj = nn.Linear(cfg.input_dim, cfg.latent_dim)
            self.encoder = nn.Sequential(
                self.token_embed,
                self.pos_embed,
                self.transformer,
                self.encoder_proj
            )
        else:
            encoder_layers = []
            in_dim = cfg.input_dim
            for hidden in cfg.hidden_dims:
                encoder_layers.append(nn.Linear(in_dim, hidden))
                encoder_layers.append(nn.ReLU())
                if cfg.dropout > 0.0:
                    encoder_layers.append(nn.Dropout(cfg.dropout))
                in_dim = hidden
            encoder_layers.append(nn.Linear(in_dim, cfg.latent_dim))
            self.encoder = nn.Sequential(*encoder_layers)

        # ---------- Decoder ----------
        decoder_layers = []
        in_dim = cfg.latent_dim
        for hidden in reversed(cfg.hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if cfg.dropout > 0.0:
                decoder_layers.append(nn.Dropout(cfg.dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, cfg.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # ---------- Optional quantum sampler ----------
        self.use_quantum_sampler = cfg.use_quantum_sampler
        if self.use_quantum_sampler:
            # The sampler simply adds Gaussian noise to the latent vector.
            self.sampler_std = cfg.sampler_std

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode the inputs into latent space."""
        return self.encoder(inputs)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors back to input space."""
        return self.decoder(latents)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        latent = self.encode(inputs)
        if self.use_quantum_sampler:
            noise = torch.randn_like(latent) * self.sampler_std
            latent = latent + noise
        return self.decode(latent)

    def reconstruct(self, inputs: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper for reconstruction."""
        return self.forward(inputs)


def train_autoencoder(model: UnifiedAutoEncoder,
                      data: torch.Tensor,
                      *,
                      epochs: int = 100,
                      batch_size: int = 64,
                      lr: float = 1e-3,
                      weight_decay: float = 0.0,
                      device: torch.device | None = None,
                      verbose: bool = False) -> list[float]:
    """Train a :class:`UnifiedAutoEncoder` and return the loss history.

    Parameters
    ----------
    model
        The autoencoder instance.
    data
        Tensor of shape (N, input_dim) containing the training data.
    epochs
        Number of training epochs.
    batch_size
        Batch size for the DataLoader.
    lr
        Learning rate for Adam optimizer.
    weight_decay
        Weight decay coefficient.
    device
        Device to run the training on.  If None, CUDA is used when available.
    verbose
        If True, prints progress.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dataset = TensorDataset(_as_tensor(data))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    history: list[float] = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(dataset)
        history.append(epoch_loss)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {epoch_loss:.6f}")
    return history


__all__ = [
    "UnifiedAutoEncoder",
    "UnifiedAutoEncoderConfig",
    "train_autoencoder",
]
