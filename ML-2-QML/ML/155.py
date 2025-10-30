import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class QuanvolutionHybrid(nn.Module):
    """
    Classical hybrid model inspired by the original Quanvolution filter.
    The model extracts 2×2 patches from a 28×28 grayscale image, applies a
    small trainable linear transformation to each patch (acting as an
    attention/feature projection), concatenates all patch embeddings,
    and feeds them into a multi‑task head that outputs both a 10‑class
    classification log‑probability and a 28×28 reconstruction map.
    """

    def __init__(self, patch_size: int = 2, num_patches: int = 14*14,
                 patch_out_dim: int = 4, num_classes: int = 10) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_out_dim = patch_out_dim
        self.num_patches = num_patches

        # Learnable projection per patch: a linear layer shared across all patches
        self.patch_proj = nn.Linear(patch_size * patch_size, patch_out_dim, bias=False)

        # Multi‑task head
        self.classifier = nn.Linear(patch_out_dim * num_patches, num_classes)
        self.reconstructor = nn.Linear(patch_out_dim * num_patches, 28 * 28)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Batch of grayscale images of shape (B, 1, 28, 28).

        Returns
        -------
        logits : torch.Tensor
            Log‑probabilities for the classification task (B, num_classes).
        recon : torch.Tensor
            Reconstructed images (B, 1, 28, 28).
        """
        B = x.size(0)
        # Extract non‑overlapping 2×2 patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches shape: (B, 1, 14, 14, 2, 2)
        patches = patches.contiguous().view(B, -1, self.patch_size * self.patch_size)
        # Apply shared projection
        patch_embeds = self.patch_proj(patches)  # (B, 196, patch_out_dim)
        features = patch_embeds.view(B, -1)      # (B, 196 * patch_out_dim)

        logits = self.classifier(features)
        recon_flat = self.reconstructor(features)
        recon = recon_flat.view(B, 1, 28, 28)

        return F.log_softmax(logits, dim=-1), recon

__all__ = ["QuanvolutionHybrid"]
