"""Classical hybrid CNN with learnable patch sampler and residual block.

This module extends the original Quanvolution architecture by replacing the fixed 2×2
patch extraction with a learnable convolutional layer and adding a residual
connection that down‑samples the input via average pooling and a 1×1 projection.
The optional `use_classifier` flag allows the module to act as a feature extractor
or as a full classifier."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class QuanvolutionHybrid(nn.Module):
    def __init__(self, num_classes: int = 10, use_classifier: bool = True):
        super().__init__()
        self.use_classifier = use_classifier
        # learnable 2×2 patch extraction
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        # residual branch: down‑sample input to 14×14 and project to 4 channels
        self.residual_pool = nn.AvgPool2d(2)
        self.residual_proj = nn.Conv2d(1, 4, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        if use_classifier:
            self.linear = nn.Linear(4 * 14 * 14, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        conv_out = self.conv(x)
        residual = self.residual_proj(self.residual_pool(x))
        out = conv_out + residual
        out = self.relu(out)
        out_flat = out.view(x.size(0), -1)
        if self.use_classifier:
            logits = self.linear(out_flat)
            return F.log_softmax(logits, dim=-1)
        else:
            return out_flat

__all__ = ["QuanvolutionHybrid"]
