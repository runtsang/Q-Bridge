import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionAdvanced(nn.Module):
    """
    Classical quanvolution-inspired network with a learnable 2×2 convolution,
    a self‑attention module to weight the extracted patches, and a linear
    classification head.  The design keeps the spirit of the original
    quanvolution filter while allowing the model to learn feature
    representations end‑to‑end.
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        attention_heads: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        # After conv we will have (H/stride) * (W/stride) patches, each of
        # size out_channels.  We treat them as a sequence for self‑attention.
        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels, num_heads=attention_heads, dropout=dropout
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        # Flattened features for classification
        self.linear = nn.Linear(out_channels * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (B, 10).
        """
        B = x.size(0)
        # Convolutional feature extraction
        features = self.conv(x)  # (B, out_channels, 14, 14)
        # Flatten spatial dimensions into a sequence
        seq = features.view(B, self.conv.out_channels, -1)  # (B, C, N)
        # Prepare for attention: (seq_len, batch, embed_dim)
        seq = seq.permute(2, 0, 1)  # (N, B, C)
        attn_out, _ = self.attention(seq, seq, seq)
        attn_out = attn_out.permute(1, 2, 0)  # (B, C, N)
        attn_out = self.norm(attn_out)
        attn_out = self.dropout(attn_out)
        # Flatten again for classification
        flat = attn_out.contiguous().view(B, -1)
        logits = self.linear(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionAdvanced"]
