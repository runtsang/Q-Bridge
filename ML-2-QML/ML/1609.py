import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionNet(nn.Module):
    """
    Classical hybrid model that extracts multi‑scale patches (2×2 and 4×4)
    and concatenates their flattened representations before a linear head.
    """
    def __init__(self, n_classes: int = 10) -> None:
        super().__init__()
        # Unfold layers for 2×2 and 4×4 patches
        self.unfold_2x2 = nn.Unfold(kernel_size=2, stride=2)
        self.unfold_4x4 = nn.Unfold(kernel_size=4, stride=4)
        # Compute feature dimensions: 4 channels per patch
        feat_2x2 = 4 * 14 * 14  # 14×14 patches from 2×2 kernel
        feat_4x4 = 4 * 7 * 7   # 7×7 patches from 4×4 kernel
        self.linear = nn.Linear(feat_2x2 + feat_4x4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (batch, 1, 28, 28).

        Returns
        -------
        torch.Tensor
            Log‑softmax logits of shape (batch, n_classes).
        """
        patches_2x2 = self.unfold_2x2(x)          # (batch, 4, 14*14)
        features_2x2 = patches_2x2.view(x.size(0), -1)
        patches_4x4 = self.unfold_4x4(x)          # (batch, 4, 7*7)
        features_4x4 = patches_4x4.view(x.size(0), -1)
        features = torch.cat([features_2x2, features_4x4], dim=1)
        logits = self.linear(features)
        return F.log_softmax(logits, dim=-1)

__all__ = ["QuanvolutionNet"]
