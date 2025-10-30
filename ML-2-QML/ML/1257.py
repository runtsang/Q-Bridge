"""Classical quanvolutional filter with multi‑head self‑attention."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuanvolutionGen203(nn.Module):
    """
    Classical quanvolutional filter that applies a 2×2 convolution followed by
    a multi‑head self‑attention layer. The final representation is fed
    into a linear classifier.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 4,
        kernel_size: int = 2,
        stride: int = 2,
        n_heads: int = 2,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=out_channels, num_heads=n_heads, batch_first=True
        )
        self.linear = nn.Linear(out_channels * 14 * 14, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, 28, 28)
        Returns:
            log‑softmax logits of shape (batch, 10)
        """
        # Convolutional feature extraction
        feat = self.conv(x)  # (batch, out_channels, 14, 14)

        # Prepare for attention: (batch, seq_len, embed_dim)
        feat = feat.view(feat.size(0), feat.size(1), -1)  # (batch, out_channels, 196)
        feat = feat.transpose(1, 2)  # (batch, 196, out_channels)

        # Self‑attention
        attn_out, _ = self.attn(feat, feat, feat)  # (batch, 196, out_channels)

        # Flatten and classify
        attn_out = attn_out.transpose(1, 2).contiguous().view(feat.size(0), -1)
        logits = self.linear(attn_out)
        return F.log_softmax(logits, dim=-1)
