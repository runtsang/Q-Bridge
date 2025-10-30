import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionGen200(nn.Module):
    """
    Classical quanvolution filter with a learnable 2×2 patch embedding
    followed by a linear classifier.  The embedding conv is replaced
    by a trainable 2×2 conv that outputs 4 channels per patch, matching
    the size used in the original quantum filter.  Dropout and batch
    norm can be enabled for regularisation.
    """
    def __init__(self,
                 in_channels: int = 1,
                 patch_size: int = 2,
                 out_channels: int = 4,
                 num_classes: int = 10,
                 dropout: float = 0.0,
                 use_batchnorm: bool = False):
        super().__init__()
        self.patch_size = patch_size
        self.embedding = nn.Conv2d(in_channels,
                                   out_channels,
                                   kernel_size=patch_size,
                                   stride=patch_size)
        self.bn = nn.BatchNorm1d(out_channels * 14 * 14) if use_batchnorm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.classifier = nn.Linear(out_channels * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract 2×2 patches, embed them with a trainable
        conv, flatten, optionally batch‑norm and dropout, then classify.
        """
        # x: (B, C, H, W) with H=W=28
        features = self.embedding(x)  # (B, out_channels, 14, 14)
        features = features.view(features.size(0), -1)  # (B, out_channels*14*14)
        if self.bn is not None:
            features = self.bn(features)
        if self.dropout is not None:
            features = self.dropout(features)
        logits = self.classifier(features)
        return F.log_softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

__all__ = ["QuanvolutionGen200"]
