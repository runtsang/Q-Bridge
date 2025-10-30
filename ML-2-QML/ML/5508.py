"""Combined classical module integrating fully connected, convolution, transformer, and self‑attention layers.

The implementation extends the original seed classes, adding a unified API that can be mixed with the
quantum counterpart.  Each sub‑component exposes a ``run`` or ``forward`` method that mirrors the
quantum interface, enabling side‑by‑side experimentation.

The main class :class:`FCLGen407` bundles the layers and offers convenience wrappers that
delegate to the appropriate sub‑module.  This design keeps the code lightweight while
providing a clear entry point for scaling experiments.

"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    "FCLGen407",
    "FullyConnectedLayer",
    "ConvFilter",
    "TransformerBlock",
    "SelfAttention",
]


class FullyConnectedLayer(nn.Module):
    """Classical fully‑connected layer that mimics the quantum example.

    Parameters
    ----------
    n_features : int, default 1
        Number of input features.
    """

    def __init__(self, n_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def run(self, thetas: np.ndarray) -> float:
        """Return the mean tanh activation over the linear projection of ``thetas``."""
        values = torch.as_tensor(thetas, dtype=torch.float32).view(-1, 1)
        return torch.tanh(self.linear(values)).mean().item()


class ConvFilter(nn.Module):
    """2‑D convolutional filter inspired by the quanvolution seed.

    Parameters
    ----------
    kernel_size : int, default 2
        Size of the square kernel.
    threshold : float, default 0.0
        Bias threshold applied before the sigmoid.
    """

    def __init__(self, kernel_size: int = 2, threshold: float = 0.0) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.threshold = threshold
        self.conv = nn.Conv2d(1, 1, kernel_size=kernel_size, bias=True)

    def run(self, data: np.ndarray) -> float:
        """Apply the convolution and return the mean sigmoid activation."""
        tensor = torch.as_tensor(data, dtype=torch.float32)
        tensor = tensor.view(1, 1, self.kernel_size, self.kernel_size)
        logits = self.conv(tensor)
        activations = torch.sigmoid(logits - self.threshold)
        return activations.mean().item()


class TransformerBlock(nn.Module):
    """Simplified transformer block composed of multi‑head attention and a feed‑forward network."""

    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard transformer block forward pass."""
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_out))


class SelfAttention:
    """Classical self‑attention helper mirroring the quantum circuit interface."""

    def __init__(self, embed_dim: int = 4) -> None:
        self.embed_dim = embed_dim

    def run(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        """Compute a self‑attention style weighted sum."""
        query = torch.as_tensor(
            inputs @ rotation_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        key = torch.as_tensor(
            inputs @ entangle_params.reshape(self.embed_dim, -1), dtype=torch.float32
        )
        value = torch.as_tensor(inputs, dtype=torch.float32)
        scores = torch.softmax(query @ key.T / np.sqrt(self.embed_dim), dim=-1)
        return (scores @ value).numpy()


class FCLGen407:
    """Unified interface that bundles the classical sub‑modules."""

    def __init__(
        self,
        n_features: int = 1,
        kernel_size: int = 2,
        embed_dim: int = 8,
        num_heads: int = 2,
        ffn_dim: int = 32,
    ) -> None:
        self.fc = FullyConnectedLayer(n_features)
        self.conv = ConvFilter(kernel_size)
        self.transformer = TransformerBlock(embed_dim, num_heads, ffn_dim)
        self.attn = SelfAttention(embed_dim)

    def run_fc(self, thetas: np.ndarray) -> float:
        return self.fc.run(thetas)

    def run_conv(self, data: np.ndarray) -> float:
        return self.conv.run(data)

    def run_transformer(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)

    def run_attention(
        self,
        rotation_params: np.ndarray,
        entangle_params: np.ndarray,
        inputs: np.ndarray,
    ) -> np.ndarray:
        return self.attn.run(rotation_params, entangle_params, inputs)
