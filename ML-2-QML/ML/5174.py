import torch
from torch import nn
import numpy as np

# Import the seed helpers
from Conv import Conv
from SelfAttention import SelfAttention
from Autoencoder import Autoencoder

class HybridSelfAttentionML(nn.Module):
    """
    Classical hybrid attention block.

    The forward pass follows the sequence:
        1. Convolutional feature extraction (ConvFilter).
        2. Self‑attention using learned rotation/entangle parameters.
        3. Optional RBF kernel similarity (via KernalAnsatz).
        4. Auto‑encoding of the attended feature map.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the embedding space for attention.
    kernel_size : int, default 2
        Size of the convolution kernel.
    latent_dim : int, default 32
        Latent dimensionality for the autoencoder.
    hidden_dims : Tuple[int, int], default (128, 64)
        Hidden layer sizes for the autoencoder.
    dropout : float, default 0.1
        Dropout probability for the autoencoder layers.
    """

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 2,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = Conv(kernel_size=kernel_size)
        self.attention = SelfAttention()
        self.autoencoder = Autoencoder(
            input_dim=embed_dim,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self.embed_dim = embed_dim

    def forward(
        self,
        inputs: torch.Tensor,
        rotation_params: torch.Tensor | None = None,
        entangle_params: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the hybrid attention pipeline.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape (..., embed_dim).
        rotation_params : torch.Tensor, optional
            Parameters for the attention rotation gates.
        entangle_params : torch.Tensor, optional
            Parameters for the attention entanglement gates.

        Returns
        -------
        torch.Tensor
            Reconstructed tensor after auto‑encoding the attended representation.
        """
        # Convolutional filtering
        conv_out = self.conv.run(inputs.detach().cpu().numpy())

        # Prepare default parameters if not supplied
        if rotation_params is None:
            rotation_params = torch.randn(self.embed_dim * 3)
        if entangle_params is None:
            entangle_params = torch.randn(self.embed_dim - 1)

        # Self‑attention
        attn_out = self.attention.run(
            rotation_params=rotation_params.detach().cpu().numpy(),
            entangle_params=entangle_params.detach().cpu().numpy(),
            inputs=conv_out,
        )

        # Convert to tensor for autoencoder
        attn_tensor = torch.as_tensor(attn_out, dtype=torch.float32)

        # Auto‑encoding
        encoded = self.autoencoder.encode(attn_tensor)
        decoded = self.autoencoder.decode(encoded)

        return decoded

__all__ = ["HybridSelfAttentionML"]
