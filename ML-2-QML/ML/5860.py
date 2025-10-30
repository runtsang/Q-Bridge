import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention__gen260(nn.Module):
    """
    Classical hybrid attention module.
    1. Convolutional front‑end (inspired by QuanvolutionFilter) extracts 2×2 patches.
    2. Multi‑head self‑attention operates on the flattened patch sequence.
    3. Optional linear head for classification (not included here, keep the module lightweight).
    """
    def __init__(self, embed_dim: int = 4, num_heads: int = 2, conv_kernel: int = 2, conv_stride: int = 2):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=conv_kernel, stride=conv_stride)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, 1, H, W), grayscale images.
        Returns:
            Tensor of shape (batch, embed_dim, H', W') after attention.
        """
        # Convolution to obtain patch embeddings
        patches = self.conv(x)          # (batch, embed_dim, H', W')
        bsz, ch, h, w = patches.shape
        seq = patches.view(bsz, ch, -1).transpose(1, 2)  # (batch, seq_len, embed_dim)

        # Multi‑head attention (self‑attention)
        attn_out, _ = self.attn(seq, seq, seq)           # (batch, seq_len, embed_dim)

        # Reshape back to spatial grid
        attn_out = attn_out.transpose(1, 2).view(bsz, ch, h, w)
        return attn_out

    def run(self, rotation_params: torch.Tensor, entangle_params: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        """
        Compatibility wrapper matching the quantum interface.
        The classical implementation ignores the rotation and entangle parameters.
        """
        return self.forward(inputs)

__all__ = ["SelfAttention__gen260"]
