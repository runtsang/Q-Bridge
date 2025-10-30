import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionGen256(nn.Module):
    """Classical quanvolution network processing 16×16 patches with residual attention.

    The network first extracts non‑overlapping 16×16 patches from a 28×28 image (after zero‑padding to 32×32),
    projects each patch to a 256‑dimensional feature vector using a depth‑wise convolution, applies
    a residual connection, then uses a multi‑head self‑attention block to capture global relationships
    across the four patches.  Finally a linear head produces class logits.
    """

    def __init__(self, num_classes: int = 10, patch_size: int = 16, embed_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Pad 28×28 image to 32×32 so that two 16×16 patches fit along each dimension.
        self.pad = nn.ZeroPad2d((0, 4, 0, 4))

        # Depth‑wise convolution that maps each 16×16 patch to a 256‑dim vector.
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # Residual connection: a simple identity for demonstration; can be replaced by a 1×1 conv.
        self.residual = nn.Identity()

        # Self‑attention across the four patches.
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # Classification head.
        self.classifier = nn.Linear(embed_dim * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (B, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑probabilities of shape (B, num_classes).
        """
        # Pad and extract patches.
        x = self.pad(x)  # (B, 1, 32, 32)
        patches = self.conv(x)  # (B, embed_dim, 2, 2)

        # Reshape to (B, 4, embed_dim)
        patches = patches.permute(0, 2, 3, 1).reshape(x.size(0), 4, self.embed_dim)

        # Residual
        patches = patches + self.residual(patches)

        # Self‑attention
        attn_out, _ = self.attn(patches, patches, patches)  # (B, 4, embed_dim)

        # Flatten and classify
        flat = attn_out.reshape(x.size(0), -1)  # (B, 4*embed_dim)
        logits = self.classifier(flat)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionGen256"]
