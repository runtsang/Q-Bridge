"""Classical self‑attention auto‑encoder combining attention and latent reconstruction.

The module exposes a single :class:`SelfAttentionAutoencoder` that first applies a
multi‑head self‑attention block, then encodes the attended representation into a
low‑dimensional latent space, and finally decodes it back to the input space.
It mirrors the interface of the original SelfAttention and Autoencoder
implementations but fuses them into a single, trainable network.

The class is fully compatible with PyTorch training loops and can be used
directly with ``torch.optim`` and ``torch.nn`` utilities.
"""

import torch
from torch import nn
from torch.nn import MultiheadAttention
from torch.utils.data import DataLoader, TensorDataset


class SelfAttentionAutoencoder(nn.Module):
    """A PyTorch model that embeds a self‑attention layer inside a classic auto‑encoder."""

    def __init__(
        self,
        embed_dim: int,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] | list[int] = (128, 64),
        dropout: float = 0.1,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim

        # Self‑attention block
        self.attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Encoder
        encoder_layers = []
        in_dim = embed_dim
        for hidden in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, hidden))
            encoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                encoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, hidden))
            decoder_layers.append(nn.ReLU())
            if dropout > 0.0:
                decoder_layers.append(nn.Dropout(dropout))
            in_dim = hidden
        decoder_layers.append(nn.Linear(in_dim, embed_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: attention → encode → decode.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, seq_len, embed_dim)``.
        """
        # Self‑attention: query, key, value are all x
        attn_output, _ = self.attn(x, x, x)
        # Flatten the sequence dimension for the encoder
        flat = attn_output.reshape(attn_output.size(0), -1)
        latent = self.encoder(flat)
        reconstructed = self.decoder(latent)
        # Reshape back to original sequence shape
        return reconstructed.reshape(x.shape)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return the latent representation of the input."""
        attn_output, _ = self.attn(x, x, x)
        flat = attn_output.reshape(attn_output.size(0), -1)
        return self.encoder(flat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent vector."""
        return self.decoder(z)


def SelfAttentionAutoencoderFactory(
    embed_dim: int,
    latent_dim: int = 32,
    hidden_dims: tuple[int, int] | list[int] = (128, 64),
    dropout: float = 0.1,
    num_heads: int = 1,
) -> SelfAttentionAutoencoder:
    """Convenience factory mirroring the original seed interface."""
    return SelfAttentionAutoencoder(embed_dim, latent_dim, hidden_dims, dropout, num_heads)


__all__ = ["SelfAttentionAutoencoder", "SelfAttentionAutoencoderFactory"]
