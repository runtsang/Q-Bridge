import torch
import torch.nn as nn
import numpy as np

class HybridSelfAttentionRegressor(nn.Module):
    """
    Classical hybrid architecture combining self‑attention, a convolutional
    quanvolution‑style feature extractor, and a quantum‑inspired fully
    connected layer. The design is inspired by the four reference seeds
    and integrates their ideas into a single end‑to‑end trainable module.
    """

    def __init__(self,
                 conv_out_channels: int = 4,
                 attention_heads: int = 1,
                 attention_dim: int = 4,
                 fc_out_features: int = 1):
        super().__init__()
        # Self‑attention over convolutional feature maps
        self.attention = nn.MultiheadAttention(embed_dim=conv_out_channels,
                                               num_heads=attention_heads,
                                               batch_first=True)
        # Quanvolution‑style convolution
        self.conv = nn.Conv2d(1, conv_out_channels, kernel_size=2, stride=2)
        # Quantum‑inspired fully‑connected layer (simulation)
        self.quantum_fc = nn.Linear(conv_out_channels, fc_out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 1, 28, 28) – e.g. MNIST images.

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch, fc_out_features).
        """
        # Convolution
        conv_out = self.conv(x)  # (batch, conv_out_channels, 14, 14)
        batch, c, h, w = conv_out.shape
        # Flatten spatial dimensions and prepare for attention
        conv_flat = conv_out.view(batch, c, h * w).permute(0, 2, 1)  # (batch, seq_len, embed_dim)
        # Self‑attention
        attn_output, _ = self.attention(conv_flat, conv_flat, conv_flat)
        # Aggregate attention output
        attn_out = attn_output.mean(dim=1)  # (batch, embed_dim)
        # Quantum‑inspired fully connected layer
        logits = self.quantum_fc(attn_out)
        return logits
