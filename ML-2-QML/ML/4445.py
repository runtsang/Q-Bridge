import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import the classical sub‑modules
from Autoencoder import Autoencoder, AutoencoderConfig
from SelfAttention import SelfAttention
from QTransformerTorch import TransformerBlockClassical

class SamplerQNNGen125(nn.Module):
    """
    Hybrid sampler that combines:
      * A lightweight classical sampler network.
      * A self‑attention module for feature weighting.
      * A transformer block for contextual representation.
      * An autoencoder for dimensionality reduction.
      * A quantum SamplerQNN (exposed via the QML module).
    """

    def __init__(
        self,
        input_dim: int = 2,
        latent_dim: int = 32,
        hidden_dims: tuple[int, int] = (128, 64),
        num_heads: int = 4,
        num_blocks: int = 2,
        ffn_dim: int = 128,
    ) -> None:
        super().__init__()
        # Classical sampler
        self.sampler_net = nn.Sequential(
            nn.Linear(input_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
        )

        # Self‑attention (classical)
        self.attention = SelfAttention()

        # Transformer block (classical)
        self.transformer = TransformerBlockClassical(
            embed_dim=4,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=0.1,
        )

        # Autoencoder
        self.autoencoder = Autoencoder(
            input_dim=4,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            dropout=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that:
          1. Generates a probability distribution via the sampler network.
          2. Embeds the probabilities into a 4‑dimensional vector.
          3. Applies self‑attention to weight the features.
          4. Feeds the weighted features through a transformer block.
          5. Encodes the transformer output with an autoencoder.
          6. Returns the concatenated latent representation and probabilities.
        """
        # Step 1: classical sampler
        probs = F.softmax(self.sampler_net(x), dim=-1)  # shape (B, 2)

        # Step 2: embed probabilities into 4‑D vector
        # Simple concatenation: [p0, p1, 1-p0, 1-p1]
        embed = torch.cat([probs, 1 - probs], dim=-1)  # shape (B, 4)

        # Step 3: self‑attention
        # SelfAttention expects rotation_params, entangle_params, inputs
        # We supply dummy parameters of appropriate shape
        rotation_params = np.zeros((4, 4))
        entangle_params = np.zeros((3,))
        attn_out = self.attention.run(rotation_params, entangle_params, embed.detach().cpu().numpy())
        attn_tensor = torch.as_tensor(attn_out, dtype=torch.float32, device=embed.device)

        # Step 4: transformer block
        # TransformerBlockClassical expects input shape (B, seq_len, embed_dim)
        # We treat each example as a single token sequence of length 1
        transformer_input = attn_tensor.unsqueeze(1)  # shape (B, 1, 4)
        transformer_out = self.transformer(transformer_input)  # shape (B, 1, 4)

        # Step 5: autoencoder encoding
        latent = self.autoencoder.encode(transformer_out.squeeze(1))  # shape (B, latent_dim)

        # Step 6: concatenate latent with probabilities
        output = torch.cat([latent, probs], dim=-1)  # shape (B, latent_dim + 2)
        return output

__all__ = ["SamplerQNNGen125"]
